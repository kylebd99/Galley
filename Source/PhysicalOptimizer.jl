using Metatheory
using Metatheory.EGraphs
include("PhysicalQueryPlan.jl")

# A recursive function which converts our logical expression tree to a phsyical plan composed of kernels.
# List of assumptions/limitations
#     - All input tensors are indexed in lexicographic order w.r.t. to index names
#     - All internal results are stored in hash tables
function expr_to_kernel(n, index_order; verbose = 0)
    kernel_root = nothing
    if n isa ReduceDim
        sub_expr = n.input
        kernel_root = AggregateExpr(n.op, n.indices, InputExpr("A",
                                                                    sub_expr.stats.indices, 
                                                                    [t_walk for _ in sub_expr.stats.indices],
                                                                    sub_expr.stats))
        input_tensors = Dict("A" => expr_to_kernel(sub_expr, index_order))
        output_indices = n.stats.indices
        output_formats = [t_hash for _ in 1:length(output_indices)]
        loop_order =  relativeSort(sub_expr.stats.indices, index_order, rev=true)
        return TensorKernel(kernel_root, n.stats, input_tensors, output_indices, output_formats, loop_order)

    elseif n isa MapJoin
        left_expr = n.left_input
        right_expr = n.right_input
        kernel_root = OperatorExpr(n.op, [InputExpr("A", 
                                            left_expr.stats.indices, 
                                                        [t_walk for _ in left_expr.stats.indices],
                                                        left_expr.stats),
                                            InputExpr("B", 
                                                right_expr.stats.indices, 
                                                        [t_walk for _ in right_expr.stats.indices],
                                                        right_expr.stats)])
        input_tensors = Dict("A" => expr_to_kernel(left_expr, index_order), "B"=>expr_to_kernel(right_expr, index_order))
        output_indices = n.stats.indices
        output_formats = [t_hash for _ in 1:length(output_indices)]
        loop_order =  relativeSort(union(left_expr.stats.indices, right_expr.stats.indices), index_order, rev=true)
        return TensorKernel(kernel_root, n.stats, input_tensors, output_indices, output_formats, loop_order)

    elseif n isa Reorder
        sub_expr = n.input
        kernel_root = ReorderExpr(n.index_order, InputExpr("A",
                                                        sub_expr.stats.indices, 
                                                        [t_walk for _ in sub_expr.stats.indices],
                                                        sub_expr.stats))
        input_tensors = Dict("A" => expr_to_kernel(sub_expr, index_order))
        output_indices = relativeSort(sub_expr.stats.indices, n.index_order)
        output_formats = [t_hash for _ in 1:length(output_indices)]
        loop_order =  relativeSort(n.stats.indices, n.index_order, rev=true)
        return TensorKernel(kernel_root, n.stats, input_tensors, output_indices, output_formats, loop_order)
    elseif n isa InputTensor
        return n.fiber
    elseif n isa Scalar
        return n.value
    end
end


