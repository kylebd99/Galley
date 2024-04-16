
# This function does a variety of sanity checks on the kernel before we attempt to execute it.
# Such as:
#  1. Check that the loop order is a permutation of the input indices
#  2. Check that the output indices are the inputs minus any that are aggregate_indices
#  3. Check that the inputs are all sorted w.r.t. the loop order

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

function check_sorted_inputs(n::PlanNode, loop_order)
    return if n.kind == Input
        @assert is_sorted_wrt_index_order([idx.name for idx in n.idxs], loop_order; loop_order=true)
    elseif n.kind == Alias
        @assert is_sorted_wrt_index_order(get_index_order(n.stats), loop_order; loop_order=true)
    elseif n.kind == Aggregate
        check_sorted_inputs(n.arg, loop_order)
    elseif n.kind == MapJoin
        for arg in n.args
            check_sorted_inputs(arg, loop_order)
        end
    elseif n.kind == Materialize
        check_sorted_inputs(n.expr, loop_order)
    end
end

function check_protocols(n::PlanNode)
    return if n.kind == Input
        @assert length(get_index_set(n.stats)) == length(get_index_protocols(n.stats))
    elseif n.kind == Alias
        @assert length(get_index_set(n.stats)) == length(get_index_protocols(n.stats))
    elseif n.kind == Aggregate
        check_protocols(n.arg)
    elseif n.kind == MapJoin
        for arg in n.args
            check_protocols(arg)
        end
    elseif n.kind == Materialize
        check_protocols(n.expr)
    end
end

function validate_physical_query(q::PlanNode)
    q = deepcopy(q)
    input_indices = get_input_indices(q.expr)
    @assert input_indices == Set([idx.name for idx in q.loop_order])
    output_indices = Set([idx.name for idx in q.expr.idx_order])
    @assert output_indices ⊆ input_indices
    @assert Set(output_indices) == Set([idx.name for idx in q.expr.idx_order])
    check_sorted_inputs(q.expr, [idx.name for idx in q.loop_order])
    check_protocols(q.expr)
end
