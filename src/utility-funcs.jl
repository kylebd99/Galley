
function initialize_tensor(formats, dims::Vector{Int64}, default_value)
    if length(dims) == 0
        return Finch.Scalar(default_value)
    end
    B = Element(default_value)
    for i in range(1, length(dims))
        if formats[i] == t_sparse_list
            B = SparseList(B, dims[i])
        elseif formats[i] == t_dense
            B = Dense(B, dims[i])
        elseif formats[i] == t_hash
            B = SparseDict(B, dims[i])
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


function get_index_order(expr, perm_choice=-1)
    if expr isa Vector{IndexExpr}
        return expr
    elseif expr isa LogicalPlanNode && perm_choice > 0
        return nthperm(sort(union([get_index_order(child) for child in expr.args]...)), perm_choice)
    elseif expr isa LogicalPlanNode
        return sort(union([get_index_order(child) for child in expr.args]...))

    else
        return []
    end
end

function fill_in_stats(expr, global_index_order)
    g = EGraph(expr)
    settermtype!(g, LogicalPlanNode)
    analyze!(g, :TensorStatsAnalysis)
    expr = e_graph_to_expr_tree(g, global_index_order)
    return expr
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
    tensor_instance = initialize_access("A", tensor, indices, [t_default for _ in indices], index_sym_dict, read=true)
    tensor_instance = call_instance(literal_instance(non_zero_func), tensor_instance)
    output_tensor = initialize_tensor([t_sparse_list for _ in indices ], [dim for dim in size(tensor)], 0.0)
    output_instance = initialize_access("output_tensor", output_tensor, indices, [t_default for _ in indices], index_sym_dict, read = false)
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
    min_size = min(length(tensor_order), length(loop_prefix))
    return reverse(tensor_order)[1:min_size] == loop_prefix[1:min_size]
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
    s_stats = TensorDef(input_indices, s)
    loop_order = reverse(input_indices)
    output_dims = [get_dim_size(s_stats, idx) for idx in output_indices]
    output_formats = [t_hash for _ in output_indices]
    if fully_compat_with_loop_prefix(output_indices, loop_order)
        output_formats = [t_sparse_list for _ in output_indices]
    end
    index_sym_dict = Dict()
    tensor_instance = initialize_access("s", s, input_indices, [t_default for _ in input_indices], index_sym_dict)
    output_tensor = initialize_tensor(output_formats, output_dims, 0.0)

    loop_index_instances = [index_instance(index_sym_dict[idx]) for idx in loop_order]
    output_variable = tag_instance(variable_instance(:output_tensor), output_tensor)
    output_access = initialize_access("output_tensor", output_tensor, output_indices, [t_default for _ in output_indices], index_sym_dict; read=false)
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
    Finch.execute(full_prgm, (mode=Finch.FastFinch(),))
    return output_tensor
end
