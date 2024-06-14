
include("annotated-query.jl")
include("greedy-optimizer.jl")
include("pruned-optimizer.jl")
include("query-splitter.jl")


function high_level_optimize(faq_optimizer::FAQ_OPTIMIZERS, q::PlanNode, ST, alias_stats, alias_hash, use_dnf)
    insert_statistics!(ST, q; bindings = alias_stats)
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
