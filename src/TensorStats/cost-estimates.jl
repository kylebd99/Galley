
# We start by defining some basic cost parameters. These will need to be adjusted somewhat
# through testing.
const SeqReadCost = 1
const SeqWriteCost = 5
const RandomReadCost = 5
const RandomWriteCost = 10

const ComputeCost = 1
const AllocateCost = 10

# We estimate the prefix cost based on the number of iterations in that prefix.
function get_loop_lookups(vars::Set{IndexExpr}, rel_conjuncts, rel_disjuncts)
    # This tensor stats doesn't actually correspond to a particular place in the expr tree,
    # so we unfortunately have to mangle the statistics interface a bit.
    join_stats = if length(rel_conjuncts) > 0
        new_def = merge_tensor_def(+, [get_def(stat) for stat in rel_conjuncts]...)
        merge_tensor_stats_join(+, new_def, rel_conjuncts...)
    else
        new_def = merge_tensor_def(+, [get_def(stat) for stat in rel_disjuncts]...)
        merge_tensor_stats_join(+, new_def, rel_disjuncts...)
    end
    lookups = estimate_nnz(join_stats, indices=vars)
    return lookups
end

# The prefix cost is equal to the number of valid iterations times the number of tensors
# which we need to access to handle that final iteration.
function get_prefix_cost(vars::Set{IndexExpr},  conjunct_stats, disjunct_stats)
    rel_conjuncts = [stat for stat in conjunct_stats if !isempty(get_index_set(stat) ∩ vars)]
    rel_disjuncts = [stat for stat in disjunct_stats if !isempty(get_index_set(stat) ∩ vars)]
    lookups = get_loop_lookups(vars, rel_conjuncts, rel_disjuncts)
    return lookups * SeqReadCost
end
