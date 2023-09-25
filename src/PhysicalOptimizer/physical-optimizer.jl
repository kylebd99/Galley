# A recursive function which converts our logical expression tree to a phsyical plan composed of kernels.
# List of assumptions/limitations
#     - All input tensors are indexed in lexicographic order w.r.t. to index names
#     - All internal results are stored in hash tables
function expr_to_kernel(n, global_index_order; verbose = 0)
    kernel_root = nothing
    if n.head == Aggregate
        op = n.args[1]
        reduce_indices = n.args[2]
        sub_expr = n.args[3]
        kernel_root = AggregateExpr(op, reduce_indices, InputExpr("A",
                                                                    sub_expr.stats.indices,
                                                                    [t_walk for _ in sub_expr.stats.indices],
                                                                    sub_expr.stats))
        input_tensors = Dict("A" => expr_to_kernel(sub_expr, global_index_order))
        output_indices = n.stats.indices
        output_formats = [t_hash for _ in 1:length(output_indices)]
        loop_order =  relative_sort(sub_expr.stats.indices, global_index_order, rev=true)
        return TensorKernel(kernel_root, n.stats, input_tensors, output_indices, output_formats, loop_order)

    elseif n.head == MapJoin
        op = n.args[1]
        left_expr = n.args[2]
        right_expr = n.args[3]
        kernel_root = OperatorExpr(op, [InputExpr("A",
                                            left_expr.stats.indices,
                                                        [t_walk for _ in left_expr.stats.indices],
                                                        left_expr.stats),
                                            InputExpr("B",
                                                right_expr.stats.indices,
                                                        [t_walk for _ in right_expr.stats.indices],
                                                        right_expr.stats)])
        input_tensors = Dict("A" => expr_to_kernel(left_expr, global_index_order), "B"=>expr_to_kernel(right_expr, global_index_order))
        output_indices = n.stats.indices
        output_formats = [t_hash for _ in 1:length(output_indices)]
        loop_order =  relative_sort(union(left_expr.stats.indices, right_expr.stats.indices), global_index_order, rev=true)
        return TensorKernel(kernel_root, n.stats, input_tensors, output_indices, output_formats, loop_order)

    elseif n.head == Reorder
        sub_expr = n.args[1]
        index_order = n.args[2]
        kernel_root = ReorderExpr(index_order, InputExpr("A",
                                                        sub_expr.stats.indices,
                                                        [t_walk for _ in sub_expr.stats.indices],
                                                        sub_expr.stats))
        input_tensors = Dict("A" => expr_to_kernel(sub_expr, global_index_order))
        output_indices = relative_sort(sub_expr.stats.indices, index_order)
        output_formats = [t_hash for _ in 1:length(output_indices)]
        loop_order =  relative_sort(n.stats.indices, index_order, rev=true)
        return TensorKernel(kernel_root, n.stats, input_tensors, output_indices, output_formats, loop_order)
    elseif n.head == InputTensor
        return n.args[2]
    elseif n.head == Scalar
        return n.args[1]
    end
end
