
include("annotated-query.jl")
include("greedy-optimizer.jl")
include("pruned-optimizer.jl")
include("query-splitter.jl")

function high_level_optimize(faq_optimizer::FAQ_OPTIMIZERS, q::PlanNode, ST, alias_stats, alias_hash, verbose)
    insert_statistics!(ST, q; bindings = alias_stats)

    # If there's the possibility of distributivity, we attempt that pushdown and see
    # whether it benefits the computation.
    check_dnf = !allequal([n.op.val for n in PostOrderDFS(q) if n.kind === MapJoin])
    logical_plan, cnf_cost = high_level_optimize(faq_optimizer, q, ST, alias_stats, alias_hash, false, verbose)
    if check_dnf
        dnf_plan, dnf_cost = high_level_optimize(faq_optimizer, q, ST, alias_stats, alias_hash, true, verbose)
        logical_plan = dnf_cost < cnf_cost ? dnf_plan : logical_plan
        verbose >= 1 && println("Used DNF: $(dnf_cost < cnf_cost)")
    end
    return logical_plan
end

function high_level_optimize(faq_optimizer::FAQ_OPTIMIZERS, q::PlanNode, ST, alias_stats, alias_hash, use_dnf, verbose)
    if faq_optimizer == greedy
        return greedy_query_to_plan(q, ST, use_dnf, alias_hash)
    elseif faq_optimizer == exact
        return exact_query_to_plan(q, ST, use_dnf, alias_hash)
    elseif faq_optimizer == pruned
        return pruned_query_to_plan(q, ST,use_dnf, alias_hash)
    elseif faq_optimizer == naive
        insert_node_ids!(q)
        return [q], Inf
    end
end
