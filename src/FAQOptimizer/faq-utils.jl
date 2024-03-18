function get_bag_stats(mult_op, sum_op, input_stats::Vector{ST}, output_indices::Set{IndexExpr}) where ST
    if length(input_stats) == 0
        return nothing
    end
    result_stats = merge_tensor_stats_join(mult_op, input_stats...)
    input_indices = get_index_set(result_stats)
    reduce_indices = setdiff(input_indices, output_indices)
    return reduce_tensor_stats(sum_op, reduce_indices, result_stats)
end
