
# We start by defining some basic cost parameters. These will need to be adjusted somewhat
# through testing.
const SeqReadCost = 1
const SeqWriteCost = 5
const RandomReadCost = 5
const RandomWriteCost = 10

const ComputeCost = 1
const AllocateCost = 10
const DenseAllocateCost = .5
const SparseAllocateCost = 60

# We estimate the prefix cost based on the number of iterations in that prefix.
function get_loop_lookups(vars::Set{IndexExpr}, rel_conjuncts, rel_disjuncts)
    # This tensor stats doesn't actually correspond to a particular place in the expr tree,
    # so we unfortunately have to mangle the statistics interface a bit.
    join_stats = if length(rel_disjuncts) == 0 || vars ⊆ union(get_index_set(stat) for stat in rel_conjuncts)
        join_def = merge_tensor_def(+, [get_def(stat) for stat in rel_conjuncts]...)
        merge_tensor_stats_join(+, join_def, rel_conjuncts...)
    elseif length(rel_conjuncts) == 0
        new_def = merge_tensor_def(+, [get_def(stat) for stat in rel_disjuncts]...)
        merge_tensor_stats_union(+, new_def, rel_disjuncts...)
    else
        disjunct_def = merge_tensor_def(+, [get_def(stat) for stat in rel_disjuncts]...)
        disjunct_stats = merge_tensor_stats_union(+, disjunct_def, rel_disjuncts...)

        join_def = merge_tensor_def(+, [get_def(stat) for stat in rel_conjuncts]..., disjunct_def)
        merge_tensor_stats_join(+, join_def, rel_conjuncts..., disjunct_stats)
    end

    lookups = estimate_nnz(join_stats, indices=vars)
    return lookups
end

# The prefix cost is equal to the number of valid iterations times the number of tensors
# which we need to access to handle that final iteration.
function get_prefix_cost(new_var, vars::Set{IndexExpr},  conjunct_stats, disjunct_stats)
    rel_conjuncts = [stat for stat in conjunct_stats if !isempty(get_index_set(stat) ∩ vars)]
    rel_disjuncts = [stat for stat in disjunct_stats if !isempty(get_index_set(stat) ∩ vars)]
    lookups = get_loop_lookups(vars, rel_conjuncts, rel_disjuncts)
    lookup_factor = 0
    for stat in union(rel_conjuncts, rel_disjuncts)
        if new_var ∉ get_index_set(stat)
            continue
        end
        if isnothing(get_index_formats(stat))
            lookup_factor += SeqReadCost
            continue
        end
        format = get_index_format(stat, new_var)
        if format == t_dense
            lookup_factor += SeqReadCost / 5
        elseif format == t_bytemap
            lookup_factor += SeqReadCost / 2.5
        else
            lookup_factor += SeqReadCost
        end
    end
    return lookups * lookup_factor
end
