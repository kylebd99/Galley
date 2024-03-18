
# We start by defining some basic cost parameters. These will need to be adjusted somewhat
# through testing.
const SeqReadCost = 1
const SeqWriteCost = 10
const RandomReadCost = 5
const RandomWriteCost = 25

const ComputeCost = 1
const AllocateCost = 25


function get_prefix_iterations(vars::Set{IndexExpr}, input_stats::Vector{TensorStats})
    prefix_stats = [stat for stat in input_stats if length(∩(get_index_set(stat), vars)) > 0]
    all_vars = union([get_index_set(stat) for stat in prefix_stats]...)

    join_stat = merge_tensor_stats_join(*, prefix_stats...)
    resulting_stat = reduce_tensor_stats(+, setdiff(all_vars, vars), join_stat)
    return estimate_nnz(resulting_stat)
end


#TODO: Remove the * and the + from this function to make it more elegant
# We estimate the prefix cost based on the number of iterations in that prefix. The tricky
# aspect of this is that intersections in the system are generally more expensive than the
# size of the smallest operand. Therefore, we need to consider the cost of extending reading
# the relevant dimension for each input tensor.
function get_loop_lookups(vars::Set{IndexExpr}, new_var::IndexExpr, input_stats::Vector{TensorStats})
    prefix_stats = [stat for stat in input_stats if length(∩(get_index_set(stat), vars)) > 0]
    iters_stat = merge_tensor_stats_join(*, prefix_stats...)
    iters_stat = reduce_tensor_stats(+, vars, iters_stat)
    lookups = estimate_nnz(iters_stat)
    return lookups
end

# The prefix cost is equal to the number of valid iterations times the number of tensors
# which we need to access to handle that final iteration.
function get_prefix_cost(vars::Set{IndexExpr}, new_var::IndexExpr, input_stats::Vector{TensorStats})
    lookups = get_loop_lookups(vars, new_var, input_stats)
    return lookups * SeqReadCost
end
