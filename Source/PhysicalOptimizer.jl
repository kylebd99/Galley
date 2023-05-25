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
        kernel_root = AggregateExpr(reduce_op, indices, InputExpr("A",
                                                                    sort(collect(sub_expr.stats.indices)), 
                                                                    [t_walk for _ in sub_expr.stats.indices],
                                                                    sub_expr.stats))
        input_tensors = Dict("A" => expr_to_kernel(sub_expr, input_tensor_dict))
        output_indices = sort(collect(n.stats.indices))
        output_formats = [t_hash for _ in 1:length(output_indices)]
        loop_order =  sort(append!(Vector(sort(output_indices, rev=true)), collect(setdiff(sub_expr.stats.indices, Set(output_indices)))), rev=true)
        return TensorKernel(kernel_root, n.stats, input_tensors, output_indices, output_formats, loop_order)

    elseif n isa MapJoin
        map_op = n.op
        left_expr = n.left_input
        right_expr = n.right_input
        kernel_root = OperatorExpr(map_op, [InputExpr("A", 
                                                        sort(collect(left_expr.stats.indices)),
                                                        [t_walk for _ in left_expr.stats.indices],
                                                        left_expr.stats),
                                            InputExpr("B", 
                                                        sort(collect(right_expr.stats.indices)),
                                                        [t_walk for _ in right_expr.stats.indices],
                                                        right_expr.stats)])
        input_tensors = Dict("A" => expr_to_kernel(left_expr, input_tensor_dict), "B"=>expr_to_kernel(right_expr, input_tensor_dict))
        output_indices = sort(collect(n.stats.indices))
        output_formats = [t_hash for _ in 1:length(output_indices)]
        loop_order =  sort(collect(union(left_expr.stats.indices, right_expr.stats.indices, Set(output_indices))), rev=true)
        return TensorKernel(kernel_root, n.stats, input_tensors, output_indices, output_formats, loop_order)

    elseif n isa InputTensor
        # Workaround to re-order tensors which are not already in lexicographic index order.
        if n.index_order != sort(n.index_order)
            map_op = +
            kernel_root = OperatorExpr(map_op, [InputExpr("A", 
                                                            sort(collect(n.stats.indices)),
                                                            [t_walk for _ in n.stats.indices],
                                                            n.stats),
                                                InputExpr("B", 
                                                            [],
                                                            [],
                                                            TensorStats(Set(), Dict(), 1, 0))])
            input_tensors = Dict("A" => input_tensor_dict[n.tensor_id], "B" => 0)
            output_indices = sort(n.index_order)
            output_formats = [t_hash for _ in 1:length(output_indices)]
            loop_order =  sort(n.index_order, rev=true)
            return TensorKernel(kernel_root, n.stats, input_tensors, output_indices, output_formats, loop_order)
        else
            return input_tensor_dict[n.tensor_id]
        end
    elseif n isa Scalar
        return n.value
    end
end


