function get_bag_stats(mult_op, sum_op, input_stats::Vector{TensorStats}, output_indices::Set{IndexExpr})
    if length(input_stats) == 0
        return nothing
    end
    result_stats = input_stats[1]
    for i in 2:length(input_stats)
        result_stats = merge_tensor_stats_join(mult_op, result_stats, input_stats[i])
    end
    input_indices = union([get_index_set(stats) for stats in input_stats]...)
    reduce_indices = setdiff(input_indices, output_indices)
    return reduce_tensor_stats(sum_op, reduce_indices, result_stats)
end
