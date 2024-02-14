#TODO: Remove the * and the + from this function to make it more elegant
# We estimate the prefix cost based on the number of valid iterations in that prefix.
function get_prefix_iterations(vars::Set{IndexExpr}, input_stats::Vector{TensorStats})
    prefix_stats = [stat for stat in input_stats if length(∩(get_index_set(stat), vars)) > 0]
    all_vars = union([get_index_set(stat) for stat in prefix_stats]...)
    resulting_stat = prefix_stats[1]
    for i in 2:length(prefix_stats)
        resulting_stat = merge_tensor_stats_join(*, resulting_stat, prefix_stats[i])
    end
    resulting_stat = reduce_tensor_stats(+, setdiff(all_vars, vars), resulting_stat)
    return estimate_nnz(resulting_stat)
end

# The prefix cost is equal to the number of valid iterations times the number of tensors
# which we need to access to handle that final iteration.
function get_prefix_cost(vars::Set{IndexExpr}, new_var::IndexExpr, input_stats::Vector{TensorStats})
    iters = get_prefix_iterations(vars, input_stats)
    intersections_per_iter = length([stat for stat in input_stats if length([new_var] ∩ get_index_set(stat)) > 0])
    return iters * intersections_per_iter
end
