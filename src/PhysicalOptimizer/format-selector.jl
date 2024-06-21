
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
        approx_sparsity = estimate_nnz(output_stats; indices=prefix) / get_dim_space_size(output_stats, Set(prefix))

        if approx_sparsity > .9
            if get_dim_space_size(output_stats, Set(prefix)) > 10^10
                throw(OutOfMemoryError())
            end
            push!(formats, t_dense)
        elseif approx_sparsity > .001
            if get_dim_space_size(output_stats, Set(prefix)) > 10^10
                throw(OutOfMemoryError())
            end
            push!(formats, t_bytemap)
        elseif needs_rw
            push!(formats, t_hash)
        else
            push!(formats, t_sparse_list)
        end
    end
    return reverse(formats)
end
