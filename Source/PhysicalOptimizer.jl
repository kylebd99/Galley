using Metatheory
using Metatheory.EGraphs
include("PhysicalQueryPlan.jl")

# A recursive function which converts our logical expression tree to a phsyical plan composed of kernels.
# List of assumptions/limitations
#     - All input tensors are indexed in lexicographic order w.r.t. to index names
#     - All internal results are stored in hash tables
function expr_to_kernel(n, input_tensor_dict; verbose = 0)
    kernel_root = nothing
    if n isa ReduceDim
        reduce_op = n.op
        indices = sort(collect(n.indices))
        sub_expr = n.input
        kernel_root = AggregateExpr(reduce_op, t_custom_agg, indices, InputExpr("A",
                                                                                sort(collect(sub_expr.stats.active_indices)), 
                                                                                [t_walk for _ in sub_expr.stats.active_indices],
                                                                                sub_expr.stats))
        input_tensors = Dict("A" => expr_to_kernel(sub_expr, input_tensor_dict))
        output_indices = sort(collect(n.stats.active_indices))
        output_formats = [t_hash for _ in 1:length(output_indices)]
        loop_order =  sort(append!(Vector(sort(output_indices, rev=true)), collect(setdiff(sub_expr.stats.active_indices, Set(output_indices)))), rev=true)
        return TensorKernel(kernel_root, n.stats, input_tensors, output_indices, output_formats, loop_order)

    elseif n isa MapJoin
        map_op = n.op
        left_expr = n.left_input
        right_expr = n.right_input
        kernel_root = OperatorExpr(map_op, t_custom_op, [InputExpr("A", 
                                                                    sort(collect(left_expr.stats.active_indices)),
                                                                    [t_walk for _ in left_expr.stats.active_indices],
                                                                    left_expr.stats),
                                                        InputExpr("B", 
                                                                    sort(collect(right_expr.stats.active_indices)),
                                                                    [t_walk for _ in right_expr.stats.active_indices],
                                                                    right_expr.stats)])
        input_tensors = Dict("A" => expr_to_kernel(left_expr, input_tensor_dict), "B"=>expr_to_kernel(right_expr, input_tensor_dict))
        output_indices = sort(collect(n.stats.active_indices))
        output_formats = [t_hash for _ in 1:length(output_indices)]
        loop_order =  sort(collect(union(left_expr.stats.active_indices, right_expr.stats.active_indices, Set(output_indices))), rev=true)
        return TensorKernel(kernel_root, n.stats, input_tensors, output_indices, output_formats, loop_order)

    elseif n isa InputTensor
        return input_tensor_dict[n.tensor_id]

    elseif n isa Scalar
        return n.value
    end
end


