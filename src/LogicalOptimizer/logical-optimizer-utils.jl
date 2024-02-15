# Return a copy of `indices` which is sorted with respect to `index_order`, e.g.
# relativeSort([a, b, c], [d, c, b, a]) = [c, b, a]
# Note: `index_order` should be a superset of `indices`
function relative_sort(indices::Vector{IndexExpr}, index_order; rev=false)
    if index_order === nothing
        return indices
    end
    sorted_indices::Vector{IndexExpr} = []
    if rev == false
        for idx in index_order
            if idx in indices
                push!(sorted_indices, idx)
            end
        end
        return sorted_indices
    else
        for idx in reverse(index_order)
            if idx in indices
                push!(sorted_indices, idx)
            end
        end
        return sorted_indices
    end
end

function relative_sort(indices::Set{IndexExpr}, index_order; rev=false)
    return relative_sort(collect(indices), index_order; rev=rev)
end

function is_sorted_wrt_index_order(indices::Vector, index_order::Vector; loop_order=false)
    if loop_order
        return issorted(indexin(indices, index_order), rev=true)
    else
        return issorted(indexin(indices, index_order))
    end
end

# This function returns the index set that occurs in the subtree of the plan rooted
# at `node`.
function get_plan_node_indices(node::LogicalPlanNode)
    if node.head == InputTensor
        return get_index_set(node.stats)
    elseif node.head == MapJoin
        return union(get_plan_node_indices(node.args[2]),
        get_plan_node_indices(node.args[3]))
    elseif node.head == Aggregate
        return setdiff(get_plan_node_indices(node.args[3]), node.args[2])
    elseif node.head == Reorder
        return get_plan_node_indices(node.args[1])
    elseif node.head == Scalar
        return Set{IndexExpr}()
    end
end
