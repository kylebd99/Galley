function get_input_stats(expr::PlanNode)
    input_stats = Dict()
    for n in PostOrderDFS(expr)
        if n.kind == Alias ||  n.kind == Input
            input_stats[n.node_id] = n.stats
        end
    end
    return input_stats
end

function reorder_input(input, expr, loop_order::Vector{IndexExpr})
    input_order = get_index_order(input.stats)
    fixed_order = relative_sort(input_order, loop_order, rev=true)
    agg_expr = Aggregate(initwrite(get_default_value(input.stats)), input)
    agg_expr.stats = input.stats
    formats = select_output_format(agg_expr.stats, reverse(input_order), fixed_order)
    reorder_query = Query(Alias(gensym("A")), Materialize(formats..., fixed_order..., agg_expr), reverse(input_order)...)
    reorder_stats = copy_stats(input.stats)
    reorder_def = get_def(reorder_stats)
    reorder_def.index_order = fixed_order
    reorder_def.level_formats = formats
    reorder_query.expr.stats = reorder_stats
    final_alias_expr = Alias(reorder_query.name.name)
    final_alias_expr.stats = copy_stats(reorder_stats)
    expr = Rewrite(Postwalk(@rule ~n => final_alias_expr where n.node_id == input.node_id))(expr)
    return [reorder_query], expr
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
function logical_query_to_physical_queries(query::PlanNode, ST, alias_stats::Dict{IndexExpr, TensorStats}; verbose = 0)
    if !(@capture query Query(~name, Materialize(~args...))) &&
            !(@capture query Query(~name, Aggregate(~args...))) &&
            !(@capture query Query(~name, MapJoin(~args...)))
        throw(ErrorException("Physical optimizer only takes in properly formatted single queries."))
    end
    insert_statistics!(ST, query, bindings=alias_stats)
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
    input_stats = get_input_stats(expr)
    for (id, stats) in input_stats
        @assert !isnothing(get_index_order(stats)) "query: $query stats: $stats"
    end
    disjuncts_and_conjuncts = get_conjunctive_and_disjunctive_inputs(expr)
    disjunct_and_conjunct_stats = (conjuncts=[s.stats for s in disjuncts_and_conjuncts.conjuncts],
                                    disjuncts=[s.stats for s in disjuncts_and_conjuncts.disjuncts])
    agg_op = isnothing(agg_op) ? initwrite(get_default_value(expr.stats)) : agg_op

    output_stats = reduce_tensor_stats(agg_op, reduce_idxs, expr.stats)
    loop_order = get_join_loop_order(disjunct_and_conjunct_stats, output_stats, output_order)
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

    needs_intermediate = length(output_order) > 0 && ((!isnothing(output_formats) && first_formats != output_formats))
    if needs_intermediate
        intermediate_query = Query(Alias(gensym("A")), Materialize(first_formats..., output_order..., expr), loop_order...)
        reorder_stats = copy_stats(expr.stats)
        reorder_def = get_def(reorder_stats)
        reorder_def.index_order = output_order
        reorder_def.level_formats = first_formats
        alias_stats[intermediate_query.name.name] = reorder_stats
        intermediate_query.expr.stats = reorder_stats
        push!(queries, intermediate_query)
        best_formats = !isnothing(output_formats) ? output_formats : select_output_format(output_stats, reverse(output_order), output_order)
        alias_expr = Alias(intermediate_query.name.name)
        alias_expr.stats = reorder_stats
        result_expr = Aggregate(initwrite(get_default_value(alias_expr.stats)), alias_expr)
        result_expr.stats = reorder_stats
        result_query = Query(query.name, Materialize(best_formats... , output_order..., result_expr), reverse(output_order)...)
        result_stats = copy_stats(reorder_stats)
        result_def = get_def(result_stats)
        result_def.index_order = output_order
        result_def.level_formats = best_formats
        result_query.expr.stats = result_stats
        push!(queries, result_query)
    else
        result_query = Query(query.name, Materialize(first_formats..., output_order..., expr), loop_order...)
        reorder_stats = copy_stats(expr.stats)
        reorder_def = get_def(reorder_stats)
        reorder_def.index_order = output_order
        reorder_def.level_formats = first_formats
        result_query.expr.stats = reorder_stats
        push!(queries, result_query)
    end
    for query in queries
        insert_node_ids!(query)
        modify_protocols!(query.expr)
        alias_stats[query.name.name] = query.expr.stats
        @assert !isnothing(get_index_order(alias_stats[query.name.name])) "$(query.name.name)"
    end
    return queries
end
