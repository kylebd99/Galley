# A recursive function which converts our logical expression tree to a phsyical plan composed of kernels.
# List of assumptions/limitations
#     - All input tensors are indexed in lexicographic order w.r.t. to index names
#     - All internal results are stored in hash tables
# TODO: Kernels should be maximal non-blocking sub-trees to reduce unecessary materialization.
# `input_counter` is a vector with length 1, so that we can pass the counter by reference.
function _recursive_get_kernel_root(n, global_index_order, input_counter)
    kernel_root = nothing
    input_dict = Dict()
    if n.head == Aggregate || n.head == Reorder
        next_tensor_id = "t_" * string(input_counter[1])
        input_counter[1] += 1
        input_dict[next_tensor_id] = expr_to_kernel(n, global_index_order)
        kernel_root = InputExpr(next_tensor_id, n.stats.indices, [t_walk for _ in n.stats.indices], n.stats)
    elseif n.head == MapJoin
        op = n.args[1]
        left_expr = n.args[2]
        right_expr = n.args[3]
        l_kernel_root, l_input_dict = _recursive_get_kernel_root(left_expr, global_index_order, input_counter)
        r_kernel_root, r_input_dict = _recursive_get_kernel_root(right_expr, global_index_order, input_counter)
        kernel_root = OperatorExpr(op, [l_kernel_root, r_kernel_root])
        input_dict = Dict(l_input_dict..., r_input_dict...)
    elseif n.head == InputTensor || n.head == Scalar
        next_tensor_id = "t_" * string(input_counter[1])
        input_counter[1] += 1
        input_dict[next_tensor_id] = n.args[2]
        kernel_root = InputExpr(next_tensor_id, n.stats.indices, [t_walk for _ in n.stats.indices], n.stats)
    end
    return kernel_root, input_dict
end


function expr_to_kernel(n, global_index_order; verbose = 0)
    kernel_root = nothing
    if n.head == Aggregate
        op = n.args[1]
        reduce_indices = n.args[2]
        sub_expr = n.args[3]
        body_kernel, input_dict = _recursive_get_kernel_root(sub_expr, global_index_order, [1])
        kernel_root = AggregateExpr(op, reduce_indices, body_kernel)
        output_indices = n.stats.indices
        output_formats = [t_hash for _ in 1:length(output_indices)]
        loop_order =  relative_sort(sub_expr.stats.indices, global_index_order, rev=true)
        return TensorKernel(kernel_root, n.stats, input_dict, output_indices, output_formats, loop_order)

    elseif n.head == MapJoin
        op = n.args[1]
        left_expr = n.args[2]
        right_expr = n.args[3]
        input_counter = [1]
        l_body_kernel, l_input_dict = _recursive_get_kernel_root(left_expr, global_index_order, input_counter)
        r_body_kernel, r_input_dict = _recursive_get_kernel_root(right_expr, global_index_order, input_counter)
        kernel_root = OperatorExpr(op, [l_body_kernel, r_body_kernel])
        input_dict = Dict(l_input_dict..., r_input_dict...)
        output_indices = n.stats.indices
        output_formats = [t_hash for _ in 1:length(output_indices)]
        loop_order =  relative_sort(output_indices, global_index_order, rev=true)
        return TensorKernel(kernel_root, n.stats, input_dict, output_indices, output_formats, loop_order)

    elseif n.head == Reorder
        sub_expr = n.args[1]
        index_order = n.args[2]
        body_kernel, input_dict = _recursive_get_kernel_root(sub_expr, global_index_order, [1])
        kernel_root = ReorderExpr(index_order, body_kernel)
        output_indices = relative_sort(sub_expr.stats.indices, index_order)
        output_formats = [t_hash for _ in 1:length(output_indices)]
        loop_order =  reverse(sub_expr.stats.indices) # We iterate in the input tensor's order for efficient reordering
        return TensorKernel(kernel_root, n.stats, input_dict, output_indices, output_formats, loop_order)
    elseif n.head == InputTensor
        return n.args[2]
    elseif n.head == Scalar
        return n.args[1]
    end
end
