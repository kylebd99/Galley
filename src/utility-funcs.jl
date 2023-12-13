using Finch: Element, SparseList, Dense, SparseHashLevel, SparseCOO
using Random

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
        return x == default_value ? 0 : 1
    end
    indices = [IndexExpr("t_" * string(i)) for i in 1:length(size(fiber))]
    fiber_instance = [initialize_access("A", fiber, indices, [t_walk for _ in indices])]
    expr_instance = @finch_program_instance non_zero_func(fiber_instance...)
    output_fiber = initialize_tensor([t_sparse_list for _ in indices ], [dim for dim in size(fiber)], 0.0)
    index_instances = [index_instance(Symbol(idx)) for idx in indices]
    full_prgm = @finch_program_instance output_fiber[index_instances...] = $expr_instance
    for index in index_instances
        full_prgm = @finch_program_instance (for $index = _; $full_prgm end)
    end
    full_prgm = @finch_program_instance (output_fiber .= 0.0 ; $full_prgm)
    return Finch.execute(full_prgm).output_fiber
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
    s_stats = NaiveStats(input_indices, s)
    output_dims = [get_dim_size(s_stats, idx) for idx in output_indices]
    output_formats = [t_hash for _ in output_indices]
    loop_order = collect(reverse(input_indices))
    if is_prefix(output_indices, loop_order)
        output_formats = [t_sparse_list for _ in output_indices]
    end

    fiber_instance = initialize_access("s", s, input_indices, [t_walk for _ in input_indices])
    output_fiber = initialize_tensor(output_formats, output_dims, 0.0)

    input_index_instances = [index_instance(Symbol(idx)) for idx in input_indices]
    output_index_instances = [index_instance(Symbol(idx)) for idx in output_indices]
    loop_index_instances = [index_instance(Symbol(idx)) for idx in input_indices]
    full_prgm = nothing
    if op == +
        full_prgm = @finch_program_instance output_fiber[output_index_instances...] += $fiber_instance
    elseif op == max
        full_prgm = @finch_program_instance output_fiber[output_index_instances...] <<max>>= $fiber_instance
    end

    for index in loop_index_instances
        full_prgm = @finch_program_instance (for $index = _; $full_prgm end)
    end
    full_prgm = @finch_program_instance (output_fiber .= 0.0 ; $full_prgm)
    println("Type of PROGRAM: ")

    display(Finch.virtualize(:unreachable, typeof(full_prgm), Finch.JuliaContext()))
    println("Type of Fiber: ", typeof(s))
    println("Size of Fiber: ", countstored(s))
    println("Input Indexes: ", input_index_instances)
    println("Output Indexes: ", output_index_instances)
    println(Finch.execute_code(full_prgm, typeof(full_prgm))|> Finch.pretty |> Finch.dataflow |> Finch.unresolve |> Finch.unquote_literals)
    return Finch.execute(full_prgm).output_fiber
end

function Base.show(io::IO ,fiber::Fiber)
    println(io, "FIBER Size(", countstored(fiber), ") Type( ", typeof(fiber), ")")
end

function Base.print(io::IO ,fiber::Fiber)
    println(io, "FIBER Size(", countstored(fiber), ") Type( ", typeof(fiber), ")")
end
