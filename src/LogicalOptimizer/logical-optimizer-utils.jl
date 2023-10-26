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

function is_sorted_wrt_index_order(indices::Vector, index_order::Vector; loop_order=false)
    if loop_order
        return issorted(indexin(indices, index_order), rev=true)
    else
        return issorted(indexin(indices, index_order))
    end
end
