#TODO: Remove the * and the + from this function to make it more elegant
function get_prefix_cost(vars::Set{IndexExpr}, input_stats::Vector{TensorStats}, prev_cost)
    prefix_stats = [stat for stat in input_stats if length(∩(get_index_set(stat), vars)) > 0]
    all_vars = union([get_index_set(stat) for stat in prefix_stats]...)
    resulting_stat = prefix_stats[1]
    for i in 2:length(prefix_stats)
        resulting_stat = merge_tensor_stats_join(*, resulting_stat, prefix_stats[i])
    end
    resulting_stat = reduce_tensor_stats(+, setdiff(all_vars, vars),resulting_stat)
    return resulting_stat.cardinality + prev_cost
end
