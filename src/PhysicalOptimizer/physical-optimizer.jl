# A recursive function which converts our logical expression tree to a phsyical plan composed of kernels.
# List of assumptions/limitations
#     - All internal results are stored in hash tables
# `input_counter` is a vector with length 1, so that we can pass the counter by reference.
# We use this counter to uniquely name the input tensors.
function _recursive_get_kernel_root(n, loop_order, input_counter)
    kernel_root = nothing
    input_dict = Dict()
    if n.head == Aggregate || n.head == Reorder
        next_tensor_id = "t_" * string(input_counter[1])
        input_counter[1] += 1
        input_indices = relative_sort(n.stats.indices, reverse(loop_order))
        input_dict[next_tensor_id] = expr_to_kernel(n, input_indices)
        stats = TensorStats(input_indices, n.stats.dim_size, n.stats.cardinality, n.stats.default_value)
        kernel_root = InputExpr(next_tensor_id, input_indices, [t_walk for _ in input_indices], stats)

    elseif n.head == MapJoin
        op = n.args[1]
        left_expr = n.args[2]
        right_expr = n.args[3]
        l_kernel_root, l_input_dict = _recursive_get_kernel_root(left_expr, loop_order, input_counter)
        r_kernel_root, r_input_dict = _recursive_get_kernel_root(right_expr, loop_order, input_counter)
        kernel_root = OperatorExpr(op, [l_kernel_root, r_kernel_root])
        input_dict = Dict(l_input_dict..., r_input_dict...)

    elseif n.head == InputTensor || n.head == Scalar
        next_tensor_id = "t_" * string(input_counter[1])
        input_counter[1] += 1
        input_dict[next_tensor_id] = transpose_input(loop_order, n.args[2],  n.stats)
        input_indices = relative_sort(Vector{IndexExpr}(n.args[1]), reverse(loop_order))
        stats = TensorStats(input_indices, n.stats.dim_size, n.stats.cardinality, n.stats.default_value)
        kernel_root = InputExpr(next_tensor_id, input_indices, [t_walk for _ in stats.indices], stats)
    end
    return kernel_root, input_dict
end

function _recursive_get_stats(n, input_counter)
    input_stats = Dict()
    if n.head == Aggregate || n.head == Reorder
        next_tensor_id = "t_" * string(input_counter[1])
        input_counter[1] += 1
        input_stats[next_tensor_id] = n.stats
    elseif n.head == MapJoin
        left_expr = n.args[2]
        right_expr = n.args[3]
        l_input_stats = _recursive_get_stats(left_expr, input_counter)
        r_input_stats = _recursive_get_stats(right_expr, input_counter)
        input_stats = Dict(l_input_stats..., r_input_stats...)
    elseif n.head == InputTensor || n.head == Scalar
        next_tensor_id = "t_" * string(input_counter[1])
        input_counter[1] += 1
        input_stats[next_tensor_id] = n.stats
    end
    return input_stats
end

# This function takes in a dict of (tensor_id => tensor_stats) and outputs a join order.
# Currently, it uses a simple prefix-join heuristic, but in the future it will be cost-based.
# TODO: Make a cost-based implementation of this function.
function get_join_loop_order(input_stats)
    num_occurrences = counter(IndexExpr)
    for stats in values(input_stats)
        for v in stats.indices
            inc!(num_occurrences, v)
        end
    end
    vars_and_counts = sort([(num_occurrences[v], v) for v in keys(num_occurrences)], by=(x)->x[1], rev=true)
    vars = [x[2] for x in vars_and_counts]
    return vars
end

# This function takes in an input and replaces it with an input expression which matches the
# loop order of the kernel. This is essentially the same as building an index on the fly
# when needed.
function transpose_input(loop_order, input, stats)
    storage_index_order = collect(reverse(loop_order))
    is_sorted = is_sorted_wrt_index_order(stats.indices, storage_index_order)
    if !is_sorted
        if input isa TensorKernel
            input.output_indices = relative_sort(input.output_indices, storage_index_order)
        else
            expr = InputExpr("t_1", stats.indices, [t_walk for _ in stats.indices], stats)
            input_indices = stats.indices
            input_dict = Dict()
            input_dict["t_1"] = input
            stats = TensorStats(relative_sort(stats.indices, storage_index_order), stats.dim_size, stats.cardinality, stats.default_value, storage_index_order)
            expr = ReorderExpr(relative_sort(stats.indices, storage_index_order), expr)
            output_formats = [t_hash for _ in 1:length(stats.indices)]
            input = TensorKernel(expr, stats, input_dict, stats.indices, output_formats, reverse(input_indices))
        end
    end
    return input
end

# This function takes in a logical plan and outputs a tree of tensor kernels.
# These kernels fully define the program that Finch will compile:
#   - Loop Order
#   - Output Format
#   - Access Protocols
# Currently, we decide the loop order entirely based on the structural properties of the
# query. Later on, we will use statistics and cost-based optimization for these decisions.
# Note, that we set the output order of the child to be the loop order of the parent.
function expr_to_kernel(n::LogicalPlanNode, output_order; verbose = 0)
    kernel_root = nothing
    if n.head == Aggregate
        op = n.args[1]
        reduce_indices = n.args[2]
        sub_expr = n.args[3]
        input_stats = _recursive_get_stats(sub_expr, [1])
        loop_order = get_join_loop_order(input_stats)
        output_indices = relative_sort(n.stats.indices, output_order)
        body_kernel, input_dict = _recursive_get_kernel_root(sub_expr, loop_order, [1])
        kernel_root = AggregateExpr(op, reduce_indices, body_kernel)
        output_formats = [t_hash for _ in 1:length(output_indices)]
        return TensorKernel(kernel_root, n.stats, input_dict, output_indices, output_formats, loop_order)

    elseif n.head == MapJoin
        op = n.args[1]
        left_expr = n.args[2]
        right_expr = n.args[3]
        input_counter = [1]
        l_input_stats = _recursive_get_stats(left_expr, input_counter)
        r_input_stats = _recursive_get_stats(right_expr, input_counter)
        input_stats = Dict(l_input_stats..., r_input_stats...)
        loop_order = get_join_loop_order(input_stats)
        output_indices = relative_sort(n.stats.indices, output_order)
        l_body_kernel, l_input_dict = _recursive_get_kernel_root(left_expr, loop_order, input_counter)
        r_body_kernel, r_input_dict = _recursive_get_kernel_root(right_expr, loop_order, input_counter)
        kernel_root = OperatorExpr(op, [l_body_kernel, r_body_kernel])
        input_dict = Dict(l_input_dict..., r_input_dict...)
        output_formats = [t_hash for _ in 1:length(output_indices)]
        return TensorKernel(kernel_root, n.stats, input_dict, output_indices, output_formats, loop_order)

    elseif n.head == Reorder
        sub_expr = n.args[1]
        output_indices = n.args[2]
        loop_order = reverse(output_order) # We iterate in the input tensor's order for efficient reordering
        body_kernel, input_dict = _recursive_get_kernel_root(sub_expr, loop_order, [1])
        kernel_root = ReorderExpr(output_indices, body_kernel)
        output_formats = [t_hash for _ in 1:length(output_indices)]
        return TensorKernel(kernel_root, n.stats, input_dict, output_indices, output_formats, loop_order)

    elseif n.head == InputTensor
        return n.args[2]

    elseif n.head == Scalar
        return n.args[1]
    end
end
