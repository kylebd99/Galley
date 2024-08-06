function branch_and_bound(input_aq::AnnotatedQuery, k, max_cost, alias_hash, cost_cache = Dict())
    input_aq = copy_aq(input_aq)
    PLAN_AND_COST = Tuple{Vector{PlanNode}, Vector{PlanNode}, AnnotatedQuery, Float64}
    optimal_orders = Dict{Set{IndexExpr}, PLAN_AND_COST}(Set{IndexExpr}()=>(PlanNode[], PlanNode[], input_aq, 0))
    prev_new_optimal_orders = optimal_orders
    # To speed up inference, we cache cost calculations for each set of already reduced idxs
    # and proposed reduction index.
    for _ in 1:length(input_aq.reduce_idxs)
        best_idx_ext = Dict()
        for (vars, pc) in prev_new_optimal_orders
            aq = pc[3]
            prev_cost = pc[4]
            for idx in get_reducible_idxs(aq)
                cost, reduced_vars = cost_of_reduce(idx, aq, cost_cache, alias_hash)
                cost += prev_cost
                new_vars = union(vars, [i.name for i in reduced_vars])
                cheapest_cost = min(get(best_idx_ext, new_vars, (nothing, nothing, nothing, nothing, Inf))[5],
                                    get(optimal_orders, new_vars, (nothing, nothing, nothing, Inf))[4],
                                    max_cost)
                if cost <= cheapest_cost + 1 # We add 1 to avoid FP issues
                    best_idx_ext[new_vars] = (aq, idx, pc[1], pc[2], cost)
                end
            end
        end
        if length(best_idx_ext) == 0
            break
        end
        new_optimal_orders = Dict{Set{IndexExpr}, PLAN_AND_COST}()
        for (new_vars, idx_ext_info) in best_idx_ext
            aq, idx, old_order, old_queries, cost = idx_ext_info
            # We don't need to recompute the resulting AQ if we already have ~some~ order
            # for that set of variables.
            query, new_aq = if haskey(optimal_orders, new_vars)
                pc = optimal_orders[new_vars]
                get_reduce_query(idx, aq), copy_aq(pc[3])
            else
                aq_copy = copy_aq(aq)
                reduce_query = reduce_idx!(idx, aq_copy)
                alias_hash[reduce_query.name] = cannonical_hash(reduce_query.expr, alias_hash)
                reduce_query, aq_copy
            end
            new_queries = PlanNode[old_queries..., query]
            new_order = PlanNode[old_order..., idx]
            new_optimal_orders[new_vars] = (new_order, new_queries, new_aq, cost)
        end

        merge!(optimal_orders, new_optimal_orders)
        # At each step, we only keep the k cheapest plans for each # of reduced idxs
        prev_new_optimal_orders = Dict()
        for i in 1:length(input_aq.reduce_idxs)
            i_length = filter((v_p)->length(v_p[1])==i, new_optimal_orders)
            num_to_keep = Int(min(k, length(i_length)))
            merge!(prev_new_optimal_orders, Dict(sort(collect(i_length), by=(v_p)->v_p[2][4])[1:num_to_keep]))
        end
    end
    return optimal_orders[Set([i.name for i in input_aq.reduce_idxs])], cost_cache
end

function pruned_query_to_plan(input_aq::AnnotatedQuery, cost_cache, alias_hash)
    greedy_queries, greedy_cost, cost_cache = greedy_query_to_plan(input_aq, cost_cache, alias_hash)
    (exact_order, exact_queries, exact_aq, exact_cost), cost_cache = branch_and_bound(input_aq, 1, greedy_cost, alias_hash, cost_cache)
    remaining_q = get_remaining_query(exact_aq)
    if !isnothing(remaining_q)
        push!(exact_queries, remaining_q)
    end
    last_query = exact_queries[end]
    # The final query should produce the result, so we ensure that it has the result's name
    last_query.name = exact_aq.output_name
    last_query.expr = Materialize(exact_aq.output_format..., exact_aq.output_order..., last_query.expr)
    last_query.expr.stats = last_query.expr.expr.stats
    get_def(last_query.expr.stats).index_order = [idx.name for idx in exact_aq.output_order]
    get_def(last_query.expr.stats).level_formats = [f.val for f in exact_aq.output_format]
    alias_hash[last_query.name] = cannonical_hash(last_query.expr, alias_hash)
    return exact_queries, exact_cost, cost_cache
end

function exact_query_to_plan(input_aq::AnnotatedQuery, cost_cache, alias_hash)
    (exact_order, exact_queries, exact_aq, exact_cost), cost_cache = branch_and_bound(input_aq,Inf, Inf, alias_hash, cost_cache)
    remaining_q = get_remaining_query(exact_aq)
    if !isnothing(remaining_q)
        push!(exact_queries, remaining_q)
    end
    last_query = exact_queries[end]
    # The final query should produce the result, so we ensure that it has the result's name
    last_query.name = exact_aq.output_name
    last_query.expr = Materialize(exact_aq.output_format..., exact_aq.output_order..., last_query.expr)
    last_query.expr.stats = last_query.expr.expr.stats
    get_def(last_query.expr.stats).index_order = [idx.name for idx in exact_aq.output_order]
    get_def(last_query.expr.stats).level_formats = [f.val for f in exact_aq.output_format]
    alias_hash[last_query.name] = cannonical_hash(last_query.expr, alias_hash)
    return exact_queries, exact_cost, cost_cache
end
