
# We start by defining some basic cost parameters. These will need to be adjusted somewhat
# through testing.
const SeqReadCost = 1
const SeqWriteCost = 10
const RandomReadCost = 5
const RandomWriteCost = 25

const ComputeCost = 1
const AllocateCost = 25

#TODO: Remove the + from this function to make it more elegant
# We estimate the prefix cost based on the number of iterations in that prefix.
function get_loop_lookups(vars::Set{IndexExpr}, join_stats::TensorStats)
    reduce_vars = setdiff(get_index_set(join_stats), vars)
    iters_stat = reduce_tensor_stats(+, reduce_vars, join_stats)
    lookups = estimate_nnz(iters_stat)
    return lookups
end

# The prefix cost is equal to the number of valid iterations times the number of tensors
# which we need to access to handle that final iteration.
function get_prefix_cost(vars::Set{IndexExpr}, join_stats::TensorStats)
    lookups = get_loop_lookups(vars, join_stats)
    return lookups * SeqReadCost
end
