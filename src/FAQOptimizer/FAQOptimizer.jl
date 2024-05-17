
include("annotated-query.jl")
include("greedy-optimizer.jl")
include("query-splitter.jl")


function high_level_optimize(faq_optimizer::FAQ_OPTIMIZERS, q::PlanNode, ST)
    insert_statistics!(ST, q)
    if faq_optimizer == greedy
        return greedy_query_to_plan(q, ST)
    elseif faq_optimizer == naive
        insert_node_ids!(q)
        return Plan(q, q.name)
    end
end
