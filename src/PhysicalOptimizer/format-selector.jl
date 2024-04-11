
function select_output_format(output_stats::TensorStats,
                                loop_order::Vector{IndexExpr},
                                output_indices::Vector{IndexExpr})
    if length(output_indices) == 0
        return LevelFormat[]
    end

    approx_sparsity = estimate_nnz(output_stats) / get_dim_space_size(get_def(output_stats), get_index_set(output_stats))
    if approx_sparsity > .1
        return [t_dense for _ in output_indices]
    end

    formats = if fully_compat_with_loop_prefix(output_indices, loop_order)
        [t_sparse_list for _ in output_indices]
    else
        [t_hash for _ in output_indices]
    end

    if length(formats) > 1
        formats[length(formats)] = t_dense
    end
    return formats
end



function select_output_format(output_stats::TensorStats,
                                loop_order::Vector{IndexExpr},
                                output_indices::Set{IndexExpr})
    if length(output_indices) == 0
        return LevelFormat[]
    end

    approx_sparsity = estimate_nnz(output_stats) / get_dim_space_size(get_def(output_stats), get_index_set(output_stats))
    if approx_sparsity > .1
        return [t_dense for _ in output_indices]
    end

    formats = if set_compat_with_loop_prefix(output_indices, loop_order)
        [t_sparse_list for _ in output_indices]
    else
        [t_hash for _ in output_indices]
    end

    if length(formats) > 1
        formats[length(formats)] = t_dense
    end
    return formats
end
