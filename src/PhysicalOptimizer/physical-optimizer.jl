function get_input_stats(alias_stats::Dict{PlanNode, TensorStats}, expr::PlanNode)
    input_stats = Dict()
    for n in PostOrderDFS(expr)
        if n.kind == Alias
            n.stats = deepcopy(alias_stats[n])
            input_stats[n.node_id] = n.stats
        elseif n.kind == Input
            input_stats[n.node_id] = n.stats
        end
    end
    return input_stats
end

function get_aliases(expr::PlanNode)
    aliases = []
    for n in PostOrderDFS(expr)
        if n.kind == Alias
            push!(aliases, n)
        end
    end
    return aliases
end

function reorder_input(input, expr, loop_order::Vector{IndexExpr})
    input_order = get_index_order(input.stats)
    fixed_order = relative_sort(input_order, loop_order, rev=true)
    agg_expr = Aggregate(initwrite(get_default_value(input.stats)), input)
    agg_expr.stats = input.stats
    formats = select_output_format(agg_expr.stats, reverse(input_order), fixed_order)
    reorder_query = Query(Alias(gensym("A")), Materialize(formats..., fixed_order..., agg_expr), reverse(input_order)...)
    reorder_stats = deepcopy(input.stats)
    reorder_def = get_def(reorder_stats)
    reorder_def.index_order = fixed_order
    reorder_def.level_formats = formats
    reorder_query.expr.stats = reorder_stats
    if formats == select_output_format(agg_expr.stats, reverse(fixed_order), fixed_order)
        final_alias_expr = Alias(reorder_query.name.name)
        final_alias_expr.stats = deepcopy(reorder_stats)
        expr = Rewrite(Postwalk(@rule ~n => final_alias_expr where n.node_id == input.node_id))(expr)
        return [reorder_query], expr
    end

    fixed_formats = select_output_format(reorder_stats, reverse(fixed_order), fixed_order)
    alias_expr = Alias(reorder_query.name.name)
    alias_expr.stats = deepcopy(reorder_stats)
    alias_agg_expr = Aggregate(initwrite(get_default_value(alias_expr.stats)), alias_expr)
    alias_agg_expr.stats = alias_expr.stats
    reformat_query = Query(Alias(gensym("A")), Materialize(fixed_formats..., fixed_order..., alias_agg_expr), reverse(fixed_order)... )
    reformat_stats = deepcopy(alias_expr.stats)
    reformat_def = get_def(reformat_stats)
    reformat_def.level_formats = fixed_formats
    reformat_query.expr.stats = reformat_stats

    final_alias_expr = Alias(reformat_query.name.name)
    final_alias_expr.stats = deepcopy(reformat_stats)
    expr = Rewrite(Postwalk(@rule ~n => final_alias_expr where n.node_id == input.node_id))(expr)
    return [reorder_query, reformat_query], expr
end


# This function takes in a query and outputs one or more queries
# The input query should take the form:
#       Query(name, Materialize(formats..., index_order..., expr))
#       or
#       Query(name, Aggregate(op, idxs..., map_expr))
#       or
#       Query(name, map_expr)
# The output queries should all be of the form:
#       Query(name, Materialize(formats..., output_order..., expr), loop_order...)
# where formats can't be t_undef.
# `alias_stats` is a dictionary which holds stats objects for the results of any previous
# queries. This is needed to get the stats for `Alias` inputs.
function logical_query_to_physical_queries(alias_stats::Dict{PlanNode, TensorStats}, query::PlanNode; verbose = 0)
    if !(@capture query Query(~name, Materialize(~args...))) &&
            !(@capture query Query(~name, Aggregate(~args...))) &&
            !(@capture query Query(~name, MapJoin(~args...)))
        throw(ErrorException("Physical optimizer only takes in properly formatted single queries."))
    end
    insert_node_ids!(query)
    id_to_node = Dict()
    for node in PreOrderDFS(query)
        id_to_node[node.node_id] = node
    end
    expr = query.expr
    output_formats = nothing
    output_order = nothing
    if expr.kind == Materialize
        if !any([f.val == t_undef for f in expr.formats])
            output_formats = [f.val for f in expr.formats]
        end
        output_order = IndexExpr[idx.name for idx in expr.idx_order]
        expr = expr.expr
    end

    agg_op = nothing
    reduce_idxs = Set{IndexExpr}()
    if expr.kind == Aggregate
        agg_op = expr.op
        reduce_idxs = Set{IndexExpr}([i.name for i in expr.idxs])
        expr = expr.arg
    end

    # Determine the optimal loop order for the query
    input_stats = get_input_stats(alias_stats, expr)
    agg_op = isnothing(agg_op) ? initwrite(get_default_value(expr.stats)) : agg_op

    output_stats = reduce_tensor_stats(agg_op, reduce_idxs, expr.stats)
    loop_order = get_join_loop_order(agg_op, Vector{TensorStats}(collect(values(input_stats))), expr.stats, output_stats, output_order)
    queries = []
    for (id, stats) in input_stats
        if !is_sorted_wrt_index_order(get_index_order(stats), loop_order; loop_order=true)
            reorder_queries, expr = reorder_input(id_to_node[id], expr, loop_order)
            append!(queries, reorder_queries)
        end
    end

    # Determine the optimal output format & add a further query to reformat if necessary.
    output_order = isnothing(output_order) ? relative_sort(get_index_set(output_stats), loop_order, rev=true) : output_order
    first_formats = select_output_format(output_stats, loop_order, output_order)

    expr = Aggregate(agg_op, reduce_idxs..., expr)
    expr.stats = output_stats

    needs_intermediate = length(output_order) > 0 && (any([f == t_hash for f in first_formats]) || (!isnothing(output_formats) && first_formats != output_formats))
    if needs_intermediate
        intermediate_query = Query(Alias(gensym("A")), Materialize(first_formats..., output_order..., expr), loop_order...)
        reorder_stats = deepcopy(expr.stats)
        reorder_def = get_def(reorder_stats)
        reorder_def.index_order = output_order
        reorder_def.level_formats = first_formats
        alias_stats[intermediate_query.name] = reorder_stats
        intermediate_query.expr.stats = reorder_stats
        push!(queries, intermediate_query)
        best_formats = !isnothing(output_formats) ? output_formats : select_output_format(output_stats, reverse(output_order), output_order)
        alias_expr = Alias(intermediate_query.name.name)
        alias_expr.stats = reorder_stats
        result_expr = Aggregate(initwrite(get_default_value(alias_expr.stats)), alias_expr)
        result_expr.stats = reorder_stats
        result_query = Query(query.name, Materialize(best_formats... , output_order..., result_expr), reverse(output_order)...)
        result_stats = deepcopy(reorder_stats)
        result_def = get_def(result_stats)
        result_def.index_order = output_order
        result_def.level_formats = best_formats
        result_query.expr.stats = result_stats
        push!(queries, result_query)
    else
        result_query = Query(query.name, Materialize(first_formats..., output_order..., expr), loop_order...)
        reorder_stats = deepcopy(expr.stats)
        reorder_def = get_def(reorder_stats)
        reorder_def.index_order =  output_order
        reorder_def.level_formats = first_formats
        result_query.expr.stats = reorder_stats
        push!(queries, result_query)
    end
    for query in queries
        insert_node_ids!(query)
        alias_stats[query.name] = query.expr.stats
    end
    return queries
end
