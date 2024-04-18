
include("annotated-query.jl")
include("greedy-optimizer.jl")


function high_level_optimize(faq_optimizer::FAQ_OPTIMIZERS, q::PlanNode, ST)
    if faq_optimizer == greedy
        return greedy_query_to_plan(q, ST)
    elseif faq_optimizer == naive
        insert_statistics!(ST, q)
        insert_node_ids!(q)
        return Plan(q, q.name)
    end
end
