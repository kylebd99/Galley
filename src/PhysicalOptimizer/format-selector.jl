
function select_output_format(output_stats::TensorStats,
                                loop_order::Vector{IndexExpr},
                                output_indices::Vector{IndexExpr})
    if length(output_indices) == 0
        return LevelFormat[]
    end

    formats = []
    for i in eachindex(output_indices)
        prefix = output_indices[length(output_indices)-i+1:end]
        needs_rw = !fully_compat_with_loop_prefix(prefix, loop_order)
        # We use division here rather than `conditional_indices` because lower=more conservative
        # here.
        prev_nnz = estimate_nnz(output_stats; indices=prefix[2:end])
        new_nnz = estimate_nnz(output_stats; indices=prefix)
        approx_nnz_per = new_nnz / prev_nnz
        # We prefer to be conservative on formats so we scale down the sparsity a bit
        approx_sparsity = approx_nnz_per / get_dim_size(output_stats, prefix[1])
        dense_memory_footprint = prev_nnz * get_dim_size(output_stats, prefix[1])
        if approx_sparsity > .5 && dense_memory_footprint < 3*10^10
#            if get_dim_space_size(output_stats, Set(prefix)) > 10^10
#                throw(OutOfMemoryError())
#            end
            push!(formats, t_dense)
        elseif approx_sparsity > .05 && dense_memory_footprint < 3*10^10 && (length(formats) == 0 ? true : formats[end] != t_bytemap) # TODO: Check out finch double bytemap bug
#            if get_dim_space_size(output_stats, Set(prefix)) > 10^10
#                throw(OutOfMemoryError())
#            end
            push!(formats, t_bytemap)
        elseif needs_rw
            push!(formats, t_hash)
        else
            push!(formats, t_sparse_list)
        end
    end
    return reverse(formats)
end
