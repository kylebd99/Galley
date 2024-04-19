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

function reorder_input(input, expr, loop_order::Vector{IndexExpr})
    input_order = get_index_order(input.stats)
    fixed_order = relative_sort(input_order, loop_order, rev=true)
    agg_expr = Aggregate(initwrite(get_default_value(input.stats)), input)
    agg_expr.stats = input.stats
    reorder_query = Query(Alias(gensym("A")), Materialize([t_hash for _ in fixed_order], fixed_order..., agg_expr), reverse(input_order)...)
    reorder_stats = deepcopy(input.stats)
    reorder_def = get_def(reorder_stats)
    reorder_def.index_order = fixed_order
    reorder_def.level_formats = [t_hash for _ in fixed_order]
    reorder_query.expr.stats = reorder_stats

    fixed_formats = select_output_format(reorder_stats, reverse(fixed_order), fixed_order)
    alias_expr = Alias(reorder_query.name.name)
    alias_expr.stats = deepcopy(reorder_stats)
    alias_agg_expr = Aggregate(initwrite(get_default_value(alias_expr.stats)), alias_expr)
    alias_agg_expr.stats = alias_expr.stats
    reformat_query = Query(Alias(gensym("A")), Materialize(fixed_formats, fixed_order..., alias_agg_expr), reverse(fixed_order)... )
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
            reorder_queries, expr = reorder_input(id_to_node[id], expr, loop_order)
            append!(queries, reorder_queries)
        end
    end
    input_indices = get_input_indices(kernel.kernel_root)
    @assert  input_indices == Set(kernel.loop_order)

    function get_output_indices(n::TensorExpression)
        return if n isa InputExpr
            Set(n.input_indices)
        elseif n isa AggregateExpr
            setdiff(get_input_indices(n.input), n.aggregate_indices)
        elseif n isa OperatorExpr
            union([get_input_indices(input) for input in n.inputs]...)
        elseif n isa ReorderExpr
            get_input_indices(n.input)
        end
    end
    output_indices = get_output_indices(kernel.kernel_root)
    @assert output_indices ⊆ input_indices
    @assert length(output_indices) == length(kernel.output_formats)

    function check_sorted_inputs(n::TensorExpression)
        if n isa InputExpr
            @assert is_sorted_wrt_index_order(n.input_indices, kernel.loop_order; loop_order=true)
        elseif n isa AggregateExpr
            check_sorted_inputs(n.input)
        elseif n isa OperatorExpr
            for input in n.inputs
                check_sorted_inputs(input)
            end
        elseif n isa ReorderExpr
            check_sorted_inputs(n.input)
        end
    end
    check_sorted_inputs(kernel.kernel_root)
end
