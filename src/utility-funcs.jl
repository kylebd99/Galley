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
    if length(indices) == 0
        return true
    end
    if loop_order
        return issorted(indexin(indices, index_order), rev=true)
    else
        return issorted(indexin(indices, index_order))
    end
end

get_index_symbol(idx_num) = Symbol("v_$(idx_num)")
get_tensor_symbol(tns_num) = Symbol("t_$(tns_num)")

# This file performs the actual execution of physical query plan.
function initialize_access(tensor_id::Symbol, tensor, index_ids, protocols, index_sym_dict; read=true)
    if tensor isa Number
        return literal_instance(tensor)
    end

    mode = read ? Reader() : Updater()
    mode = literal_instance(mode)
    index_expressions = []
    for i in range(1, length(index_ids))
        if !haskey(index_sym_dict, index_ids[i])
            idx_num = length(index_sym_dict)
            index_sym_dict[index_ids[i]] = get_index_symbol(idx_num)
        end
        index = index_instance(index_sym_dict[index_ids[i]])
        if read == true
            if protocols[i] == t_walk
                index = call_instance(literal_instance(walk), index)
            elseif protocols[i] == t_gallop
                index = call_instance(literal_instance(gallop), index)
            elseif protocols[i] == t_lead
                index = call_instance(literal_instance(lead), index)
            elseif protocols[i] == t_follow
                index = call_instance(literal_instance(follow), index)
            end
        end
        push!(index_expressions, index)
    end
    tensor_var = variable_instance(tensor_id)
    tensor_tag = tag_instance(tensor_var, tensor)
    tensor_access = access_instance(tensor_tag, mode, index_expressions...)
    return tensor_access
end

function initialize_tensor(formats, dims, default_value)
    if length(dims) == 0
        return Finch.Scalar(default_value)
    end
    B = Element(default_value)
    for i in range(1, length(dims))
        if formats[i] == t_sparse_list
            B = SparseList(B, dims[i])
        elseif formats[i] == t_dense
            B = Dense(B, dims[i])
        elseif formats[i] == t_bytemap
            B = SparseByteMap(B, dims[i])
        elseif formats[i] == t_hash
            B = Sparse(B, dims[i])
        else
            println("Error: Attempted to initialize invalid level format type.")
        end
    end
    return Tensor(B)
end


# Generates a tensor whose non-default entries are distributed uniformly randomly throughout.
function uniform_tensor(shape, sparsity; formats = [], default_value = 0, non_default_value = 1)
    if formats == []
        formats = [t_sparse_list for _ in 1:length(shape)]
    end

    tensor = initialize_tensor(formats, shape, default_value)
    copyto!(tensor, fsprand(Tuple(shape), sparsity, (r, n)->[non_default_value for _ in 1:n]))
    return tensor
end

# This function takes in a tensor and outputs the 0/1 tensor which is 0 at all default
# values and 1 at all other entries.
function get_sparsity_structure(tensor::Tensor)
    default_value = Finch.default(tensor)
    function non_zero_func(x)
        return x == default_value ? 0.0 : 1.0
    end
    index_sym_dict = Dict()
    indices = [IndexExpr("t_" * string(i)) for i in 1:length(size(tensor))]
    tensor_instance = initialize_access(:A, tensor, indices, [t_default for _ in indices], index_sym_dict, read=true)
    tensor_instance = call_instance(literal_instance(non_zero_func), tensor_instance)
    output_tensor = initialize_tensor([t_sparse_list for _ in indices ], [dim for dim in size(tensor)], 0.0)
    output_instance = initialize_access(:output_tensor, output_tensor, indices, [t_default for _ in indices], index_sym_dict, read = false)
    full_prgm = assign_instance(output_instance, literal_instance(initwrite(0.0)), tensor_instance)

    for index in indices
        full_prgm = loop_instance(index_instance(index_sym_dict[index]), Dimensionless(), full_prgm)
    end

    initializer = declare_instance(variable_instance(:output_tensor), literal_instance(0.0))
    full_prgm = block_instance(initializer, full_prgm)
    Finch.execute(full_prgm)
    return output_tensor
end

function fully_compat_with_loop_prefix(tensor_order::Vector, loop_prefix::Vector)
    for i in eachindex(tensor_order)
        if i > length(loop_prefix)
            return true
        end
        if reverse(tensor_order)[i] != loop_prefix[i]
            return false
        end
    end
    return true
end

# This function determines whether any ordering of the `l_set` is a prefix of `r_vec`.
# If r_vec is smaller than l_set, we just check whether r_vec is a subset of l_set.
function set_compat_with_loop_prefix(tensor_order::Set, loop_prefix::Vector)
    if length(tensor_order) > length(loop_prefix)
        return Set(loop_prefix) ⊆ tensor_order
    else
        return tensor_order == Set(loop_prefix[1:length(tensor_order)])
    end
end

# Takes in a tensor `s` with indices `input_indices`, and outputs a tensor which has been
# contracted to only `output_indices` using the aggregation operation `op`.
function one_off_reduce(op,
                        input_indices,
                        output_indices,
                        s::Tensor)
    s_stats = TensorDef(s, input_indices)
    loop_order = []
    for i in reverse(input_indices)
        if i ∉ loop_order
            push!(loop_order, i)
        end
    end
    output_dims = [get_dim_size(s_stats, idx) for idx in output_indices]
    output_formats = [t_hash for _ in output_indices]
    if fully_compat_with_loop_prefix(output_indices, loop_order)
        output_formats = [t_sparse_list for _ in output_indices]
    end
    index_sym_dict = Dict()
    tensor_instance = initialize_access(:s, s, input_indices, [t_default for _ in input_indices], index_sym_dict)
    output_tensor = initialize_tensor(output_formats, output_dims, 0.0)
    loop_index_instances = [index_instance(index_sym_dict[idx]) for idx in loop_order]
    output_variable = tag_instance(variable_instance(:output_tensor), output_tensor)
    output_access = initialize_access(:output_tensor, output_tensor, output_indices, [t_default for _ in output_indices], index_sym_dict; read=false)
    op_instance = if op == max
        literal_instance(initmax(Finch.default(s)))
    elseif op == min
        literal_instance(initmin(Finch.default(s)))
    else
        literal_instance(op)
    end
    full_prgm = assign_instance(output_access, op_instance, tensor_instance)

    for index in reverse(loop_index_instances)
        full_prgm = loop_instance(index, Dimensionless(), full_prgm)
    end
    initializer = declare_instance(output_variable, literal_instance(0.0))
    full_prgm = block_instance(initializer, full_prgm)
    Finch.execute(full_prgm, mode=:fast)
    return output_tensor
end
