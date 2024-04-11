
function greedy_aq_to_plan(aq::AnnotatedQuery)
    queries = []
    reducible_idxs = get_reducible_idxs(aq)
    while !isempty(reducible_idxs)
        cheapest_idx = nothing
        cheapest_cost = Inf
        for idx in reducible_idxs
            cost = cost_of_reduce(idx, aq)
            if cost < cheapest_cost
                cheapest_idx = idx
                cheapest_cost = cost
            end
        end
        query, new_aq = reduce_idx(cheapest_idx, aq)
        aq = new_aq
        push!(queries, query)
        reducible_idxs = get_reducible_idxs(aq)
    end
    return Plan(queries..., Outputs(queries[end].name))
end
