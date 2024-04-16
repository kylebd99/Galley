function get_input_stats(alias_stats::Dict{PlanNode, TensorStats}, expr::PlanNode)
    input_stats = Dict()
    for n in PostOrderDFS(expr)
        if n.kind == Alias
            n.stats = alias_stats[n]
            input_stats[n.node_id] = n.stats
        elseif n.kind == Input
            input_stats[n.node_id] = n.stats
        end
    end
    return input_stats
end

function reorder_input(alias_stats, input, expr, loop_order::Vector{IndexExpr})
    input_order = get_index_order(input.stats)
    fixed_order = relative_sort(input_order, loop_order, rev=true)
    agg_expr = Aggregate(initwrite(get_default_value(input.stats)), input)
    agg_expr.stats = input.stats
    reorder_query = Query(Alias(gensym("A")), Materialize([t_hash for _ in fixed_order], fixed_order..., agg_expr), reverse(input_order)...)
    reorder_stats = deepcopy(input.stats)
    reorder_def = get_def(reorder_stats)
    reorder_def.index_order = fixed_order
    reorder_def.level_formats = [t_hash for _ in fixed_order]
    alias_stats[reorder_query.name] = reorder_stats

    fixed_formats = select_output_format(reorder_stats, reverse(fixed_order), fixed_order)
    alias_expr = Alias(reorder_query.name.name)
    alias_expr.stats = deepcopy(reorder_stats)
    alias_agg_expr = Aggregate(initwrite(get_default_value(alias_expr.stats)), alias_expr)
    alias_agg_expr.stats = alias_expr.stats
    reformat_query = Query(Alias(gensym("A")), Materialize(fixed_formats, fixed_order..., alias_agg_expr), reverse(fixed_order)... )
    reformat_stats = deepcopy(alias_expr.stats)
    reformat_def = get_def(reformat_stats)
    reformat_def.level_formats = fixed_formats
    alias_stats[reformat_query.name] = reformat_stats

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
    if !(@capture query Query(~name, Aggregate(~args...))) &&
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
        if !any([f == t_undef for f in expr.formats])
            output_formats = expr.formats
        end
        output_order = [idx.name for idx in expr.idx_order]
        expr = expr.expr
    end

    agg_op = nothing
    reduce_idxs = Set()
    if expr.kind == Aggregate
        agg_op = expr.op
        reduce_idxs = expr.idxs
        expr = expr.arg
    end

    # Determine the optimal loop order for the query
    input_stats = get_input_stats(alias_stats, expr)
    condense_stats!(expr.stats)
    output_stats = reduce_tensor_stats(agg_op, reduce_idxs, expr.stats)
    loop_order = get_join_loop_order(agg_op, Vector{TensorStats}(collect(values(input_stats))), expr.stats, output_stats, output_order)
    queries = []
    for (id, stats) in input_stats
        if !is_sorted_wrt_index_order(get_index_order(stats), loop_order; loop_order=true)
            reorder_queries, expr = reorder_input(alias_stats, id_to_node[id], expr, loop_order)
            for query in reorder_queries
                reordered_input_stats = get_input_stats(alias_stats, query)
                modify_protocols!(collect(values(reordered_input_stats)))
            end
            append!(queries, reorder_queries)
        end
    end


    # Determine the optimal access protocols for every index occurence
    reordered_input_stats = get_input_stats(alias_stats, expr)
    modify_protocols!(collect(values(reordered_input_stats)))

    # Determine the optimal output format & add a further query to reformat if necessary.
    output_order = isnothing(output_order) ? relative_sort(get_index_set(output_stats), loop_order, rev=true) : output_order
    first_formats = select_output_format(output_stats, loop_order, output_order)

    agg_op = isnothing(agg_op) ? initwrite(get_default_value(output_stats)) : agg_op
    expr = Aggregate(agg_op, reduce_idxs..., expr)
    expr.stats = output_stats

    needs_intermediate = any([f == t_hash for f in first_formats]) || !isnothing(output_formats)
    if needs_intermediate
        intermediate_query = Query(Alias(gensym("A")), Materialize(first_formats..., output_order..., expr), loop_order...)
        reorder_stats = deepcopy(expr.stats)
        reorder_def = get_def(reorder_stats)
        reorder_def.index_order = output_order
        reorder_def.level_formats = first_formats
        reorder_def.index_protocols = [t_default for _ in output_order]
        alias_stats[intermediate_query.name] = reorder_stats
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
        alias_stats[query.name] = result_stats
        push!(queries, result_query)
    else
        result_query = Query(query.name, Materialize(first_formats..., output_order..., expr), loop_order...)
        reorder_stats = deepcopy(expr.stats)
        reorder_def = get_def(reorder_stats)
        reorder_def.index_order =  output_order
        reorder_def.level_formats = first_formats
        alias_stats[query.name] = reorder_stats
        push!(queries, result_query)
    end
    return queries
end

# This function does a variety of sanity checks on the kernel before we attempt to execute it.
# Such as:
#  1. Check that the loop order is a permutation of the input indices
#  2. Check that the output indices are the inputs minus any that are aggregate_indices
#  3. Check that the inputs are all sorted w.r.t. the loop order
function validate_physical_query(q::PlanNode, alias_stats)
    q = deepcopy(q)
    insert_statistics!(NaiveStats, q, bindings = alias_stats, replace=true)
    function get_input_indices(n::PlanNode)
        return if n.kind == Input
            get_index_set(n.stats)
        elseif n.kind == Alias
            get_index_set(n.stats)
        elseif  n.kind == Aggregate
            get_input_indices(n.arg)
        elseif  n.kind == MapJoin
            union([get_input_indices(input) for input in n.args]...)
        elseif n.kind == Materialize
            get_input_indices(n.expr)
        end
    end
    input_indices = get_input_indices(q.expr)
    @assert input_indices == Set([idx.name for idx in q.loop_order])

    function get_output_indices(n::PlanNode)
        return if n.kind == Input
            get_index_set(n.stats)
        elseif n.kind == Alias
            get_index_set(n.stats)
        elseif n.kind == Aggregate
            setdiff(get_input_indices(n.arg), n.idxs)
        elseif  n.kind == MapJoin
            union([get_input_indices(input) for input in n.args]...)
        elseif n.kind == Materialize
            get_input_indices(n.expr)
        end
    end
    output_indices = get_index_set(q.expr.stats)
    @assert output_indices ⊆ input_indices
    @assert Set(output_indices) == Set([idx.name for idx in q.expr.idx_order])

    function check_sorted_inputs(n::PlanNode)
        return if n.kind == Input
            @assert is_sorted_wrt_index_order([idx.name for idx in n.idxs], [idx.name for idx in q.loop_order]; loop_order=true)
        elseif n.kind == Alias
            @assert is_sorted_wrt_index_order(get_index_order(n.stats), [idx.name for idx in q.loop_order]; loop_order=true)
        elseif n.kind == Aggregate
            check_sorted_inputs(n.arg)
        elseif n.kind == MapJoin
            for arg in n.args
                check_sorted_inputs(arg)
            end
        elseif n.kind == Materialize
            check_sorted_inputs(n.expr)
        end
    end
    check_sorted_inputs(q.expr)
end
