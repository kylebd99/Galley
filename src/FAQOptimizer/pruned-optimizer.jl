


function beam_search(input_query::PlanNode, ST, k, max_cost)
    input_aq = AnnotatedQuery(input_query, ST)
    PLAN_AND_COST = Tuple{Vector{PlanNode}, Vector{PlanNode}, AnnotatedQuery, Float64}
    optimal_orders = Dict{Set{IndexExpr}, PLAN_AND_COST}(Set{IndexExpr}()=>(PlanNode[], PlanNode[], input_aq, 0))
    for min_reduced_idxs in 1:length(input_aq.reduce_idxs)
        new_optimal_orders = Dict{Set{IndexExpr}, PLAN_AND_COST}()
        for (vars, pc) in optimal_orders
            aq = pc[3]
            prev_cost = pc[4]
            for idx in get_reducible_idxs(aq)
                cost, reduced_vars = cost_of_reduce(idx, aq)
                cost += prev_cost
                new_vars = union(vars, [i.name for i in reduced_vars])
                cheapest_cost = min(get(new_optimal_orders, new_vars, (nothing, nothing, nothing, Inf))[4],
                                    get(optimal_orders, new_vars, (nothing, nothing, nothing, Inf))[4])
                if cost <= min(max_cost, cheapest_cost)
                    new_order = [pc[1]..., idx]
                    new_aq = copy(aq)
                    query = reduce_idx!(idx, new_aq)
                    new_queries = PlanNode[pc[2]..., query]
                    new_optimal_orders[new_vars] = (new_order, new_queries, new_aq, cost)
                end
            end
        end
        filter!((x)->length(x[1]) >= min_reduced_idxs, optimal_orders)
        for i in min_reduced_idxs:length(input_aq.reduce_idxs)
            length_i_opt_orders = [x for x in new_optimal_orders if length(x[1]) == i]
            num_orders = Int(min(k, length(length_i_opt_orders)))
            merge!(optimal_orders, Dict(sort(length_i_opt_orders, by=(v_and_pc)->v_and_pc[2][4])[1:num_orders]))
        end
    end
    return only(values(optimal_orders))
end

function pruned_query_to_plan(input_query::PlanNode, ST)
    greedy_order, greedy_queries, greedy_aq, greedy_cost = beam_search(plan_copy(input_query), ST, 3, Inf)
    exact_order, exact_queries, exact_aq, exact_cost = beam_search(plan_copy(input_query), ST, Inf, greedy_cost)
    remaining_q = get_remaining_query(exact_aq)
    if !isnothing(remaining_q)
        push!(exact_queries, remaining_q)
    end
    last_query = exact_queries[end]
    last_query.expr = Materialize(exact_aq.output_format..., exact_aq.output_order..., last_query.expr)
    last_query.expr.stats = last_query.expr.expr.stats
    return Plan(exact_queries..., Outputs(last_query.name))
end
