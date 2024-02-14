
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
            B = SparseHashLevel(B, Tuple([dims[i]]))
        else
            println("Error: Attempted to initialize invalid level format type.")
        end
    end
    return Fiber!(B)
end


# Generates a fiber whose non-default entries are distributed uniformly randomly throughout.
function uniform_fiber(shape, sparsity; formats = [], default_value = 0, non_default_value = 1)
    if formats == []
        formats = [t_sparse_list for _ in 1:length(shape)]
    end

    fiber = initialize_tensor(formats, shape, default_value)
    copyto!(fiber, fsprand(Tuple(shape), sparsity, (r, n)->[non_default_value for _ in 1:n]))
    return fiber
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
function get_sparsity_structure(fiber::Fiber)
    default_value = Finch.default(fiber)
    function non_zero_func(x)
        return x == default_value ? 0.0 : 1.0
    end
    indices = [IndexExpr("t_" * string(i)) for i in 1:length(size(fiber))]
    fiber_instance = initialize_access("A", fiber, indices, [t_walk for _ in indices], read=true)
    fiber_instance = call_instance(literal_instance(non_zero_func), fiber_instance)
    output_fiber = initialize_tensor([t_sparse_list for _ in indices ], [dim for dim in size(fiber)], 0.0)
    output_instance = initialize_access("output_fiber", output_fiber, indices, [t_walk for _ in indices], read = false)
    full_prgm = assign_instance(output_instance, literal_instance(initwrite(0.0)), fiber_instance)

    for index in indices
        full_prgm = loop_instance(index_instance(Symbol(index)), Dimensionless(), full_prgm)
    end

    initializer = declare_instance(variable_instance(:output_fiber), literal_instance(0.0))
    full_prgm = block_instance(initializer, full_prgm)

    output_fiber = Finch.execute(full_prgm).output_fiber
    return output_fiber
end

function is_prefix(l_vec::Vector, r_vec::Vector)
    if length(l_vec) > length(r_vec)
        return false
    end
    for i in eachindex(l_vec)
        if l_vec[i] != r_vec[i]
            return false
        end
    end
    return true
end

# Takes in a fiber `s` with indices `input_indices`, and outputs a fiber which has been
# contracted to only `output_indices` using the aggregation operation `op`.
function one_off_reduce(op,
                        input_indices,
                        output_indices,
                        s::Fiber)
    s_stats = TensorDef(input_indices, s)
    loop_order = reverse(input_indices)
    output_dims = [get_dim_size(s_stats, idx) for idx in output_indices]
    output_formats = [t_hash for _ in output_indices]
    if is_prefix(output_indices, loop_order)
        output_formats = [t_sparse_list for _ in output_indices]
    end
    fiber_instance = initialize_access("s", s, input_indices, [t_walk for _ in input_indices])
    output_fiber = initialize_tensor(output_formats, output_dims, 0.0)

    loop_index_instances = [index_instance(Symbol(idx)) for idx in loop_order]
    output_variable = tag_instance(variable_instance(:output_fiber), output_fiber)
    output_access = initialize_access("output_fiber", output_fiber, output_indices, [t_walk for _ in output_indices]; read=false)
    op_instance = if op == max
        literal_instance(initmax(Finch.default(s)))
    elseif op == min
        literal_instance(initmin(Finch.default(s)))
    else
        literal_instance(op)
    end
    full_prgm = assign_instance(output_access, op_instance, fiber_instance)

    for index in reverse(loop_index_instances)
        full_prgm = loop_instance(index, Dimensionless(), full_prgm)
    end
    initializer = declare_instance(output_variable, literal_instance(0.0))
    full_prgm = block_instance(initializer, full_prgm)
    return Finch.execute(full_prgm, (mode=Finch.FastFinch(),)).output_fiber
end

#function Base.show(io::IO ,fiber::Fiber)
#    println(io, "FIBER Type( ", typeof(fiber), ")")
#end

function Base.print(io::IO ,fiber::Fiber)
    println(io, "FIBER Type( ", typeof(fiber), ")")
end
