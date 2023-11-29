# Before performing a full HTD, we first prune the problem to eliminate any obvious
# decisions. By "obvious", we specifically mean any choices which can only incur a linear
# overhead in the size of the input data.
# Currently these optimizations include:
# 1. Aggregate over any variables that are only included in a single factor
# 2. Perform prefix joins, i.e. joins where one factor fully covers another
# These are performed iteratively until they can no longer be applied.

function make_join_factor(op, lfactor::Factor, rfactor::Factor)
    return Factor(MapJoin(op, lfactor.input, rfactor.input),
                    union(lfactor.active_indices,rfactor.active_indices),
                    union(lfactor.all_indices, rfactor.all_indices),
                    false,
                    merge_tensor_stats_join(op, lfactor.stats, rfactor.stats))
end

function merge_prefix_joins!(faq::FAQInstance)
    made_change = false
    finished = false
    prev_factors = faq.factors
    while !finished
        new_factors = Set{Factor}()
        handled_factors = Set{Factor}()
        for factor in prev_factors
            factor in handled_factors && continue
            for sub_factor in prev_factors
                sub_factor in handled_factors && continue
                factor == sub_factor && continue
                if factor.all_indices ⊇ sub_factor.all_indices
                    push!(new_factors, make_join_factor(faq.mult_op, factor, sub_factor))
                    push!(handled_factors, factor)
                    push!(handled_factors, sub_factor)
                    break
                end
            end
        end

        for factor in prev_factors
            if !(factor in handled_factors)
                push!(new_factors, factor)
            end
        end

        if prev_factors == new_factors
            finished = true
        else
            made_change = true
        end
        prev_factors = new_factors
    end
    faq.factors = prev_factors
    return made_change
end

function aggregate_dangling_vars!(faq::FAQInstance)
    made_change = false
    indices = copy(faq.input_indices)
    for idx in indices
        rel_factors = Factor[]
        for factor in faq.factors
            if idx in factor.all_indices
                push!(rel_factors, factor)
            end
        end
        if length(rel_factors) == 1 && !(idx in faq.output_indices)
            factor = rel_factors[1]
            factor.input = Aggregate(faq.sum_op, Set{IndexExpr}([idx]), factor.input)
            delete!(factor.active_indices, idx)
            delete!(factor.all_indices, idx)
            delete!(faq.input_indices, idx)
            made_change = true
        end
    end
    return made_change
end

function prune_faq!(faq::FAQInstance)
    finished = false
    while !finished
        merged_joins = merge_prefix_joins!(faq)
        aggregated_indices = aggregate_dangling_vars!(faq)
        finished = !(merged_joins || aggregated_indices)
    end
end
