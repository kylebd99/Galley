
# This function takes in a dict of (tensor_id => tensor_stats) and outputs a join order.
# Currently, it uses a simple prefix-join heuristic, but in the future it will be cost-based.
function get_join_loop_order_simple(input_stats)
    num_occurrences = counter(IndexExpr)
    for stats in values(input_stats)
        for v in get_index_set(stats)
            inc!(num_occurrences, v)
        end
    end
    vars_and_counts = sort([(num_occurrences[v], v) for v in keys(num_occurrences)], by=(x)->x[1], rev=true)
    vars = [x[2] for x in vars_and_counts]
    return vars
end

cost_of_reformat(stat::TensorStats) = estimate_nnz(stat) * RandomWriteCost

function get_reformat_set(input_stats::Vector{TensorStats}, prefix::Vector{IndexExpr})
    ref_set = Set()
    for i in eachindex(input_stats)
        index_order = get_index_order(input_stats[i])
        # We don't need to worry about child bags when calculating reformat cost
        if isnothing(index_order)
            continue
        end
        # Tensors are stored in column major, so we reverse the index order here
        index_order = reverse(index_order)
        occurences = [isnothing(x) ? Inf : x for x  in indexin(index_order, prefix)]
        !issorted(occurences) && push!(ref_set, i)
    end
    return ref_set
end

function get_output_compat(ordered_output_vars::Vector{IndexExpr}, prefix::Vector{IndexExpr})
    return if fully_compat_with_loop_prefix(ordered_output_vars, prefix)
        FULL_PREFIX
    elseif set_compat_with_loop_prefix(Set(ordered_output_vars), prefix)
        SET_PREFIX
    else
        NO_PREFIX
    end
end

@enum OUTPUT_COMPAT FULL_PREFIX=0 SET_PREFIX=1 NO_PREFIX=2
# We use a version of Selinger's algorithm to determine the join loop ordering.
# For each subset of variables, we calculate their optimal ordering via dynamic programming.
# This is singly exponential in the number of loop variables, and it uses the statistics to
# determine the costs.
function get_join_loop_order(input_stats::Vector{TensorStats}, output_stats::TensorStats, output_order::Vector{IndexExpr})
    all_vars = union([get_index_set(stat) for stat in input_stats]...)
    output_vars = get_index_set(output_stats)
    ordered_output_vars = relative_sort(output_vars, output_order)

    # At all times, we keep track of the best plans for each level of output compatability.
    # This will let us consider the cost of random writes and transposes at the end.
    reformat_costs = Dict(i => cost_of_reformat(input_stats[i]) for i in eachindex(input_stats))
    PLAN_CLASS = Tuple{Set{IndexExpr}, OUTPUT_COMPAT, Set{Int}}
    PLAN = Tuple{Vector{IndexExpr}, Float64}
    optimal_plans = Dict{PLAN_CLASS, PLAN}()
    for var in all_vars
        prefix = [var]
        v_set = Set(prefix)
        rf_set = get_reformat_set(input_stats, prefix)
        output_compat = get_output_compat(ordered_output_vars, prefix)
        class = (v_set, output_compat, rf_set)
        cost = get_prefix_cost(v_set, var, input_stats)
        optimal_plans[class] = (prefix, cost)
    end

    for _ in 2:length(all_vars)
        new_plans =  Dict{PLAN_CLASS, PLAN}()
        for (plan_class, plan) in optimal_plans
            prefix_set = plan_class[1]
            prefix = plan[1]
            cost = plan[2]
            # We only consider extensions that don't result in cross products
            potential_vars = Set{IndexExpr}()
            for stat in input_stats
                index_set = get_index_set(stat)
                if length(∩(index_set, prefix_set)) > 0
                    potential_vars = ∪(potential_vars, index_set)
                end
            end
            potential_vars = setdiff(potential_vars, prefix_set)
            if length(potential_vars) == 0
                # If the query isn't connected, we will need to include a cross product
                potential_vars = setdiff(all_vars, prefix_set)
            end

            for new_var in potential_vars
                new_prefix_set = union(prefix_set, [new_var])
                new_prefix = [prefix..., new_var]
                rf_set = get_reformat_set(input_stats, new_prefix)
                output_compat = get_output_compat(ordered_output_vars, new_prefix)
                new_plan_class = (new_prefix_set, output_compat, rf_set)
                new_cost = get_prefix_cost(new_prefix_set, new_var, input_stats) + cost
                new_plan = (new_prefix, new_cost)

                alt_cost = Inf
                if haskey(new_plans, plan_class)
                    alt_cost = new_plans[plan_class][2]
                end

                if new_cost < alt_cost
                    new_plans[new_plan_class] = new_plan
                end
            end
        end

        plans_by_set = Dict()
        for (plan_class, plan) in new_plans
            idx_set = plan_class[1]
            if !haskey(plans_by_set, idx_set)
                plans_by_set[idx_set] = Dict()
            end
            plans_by_set[idx_set][plan_class] = plan
        end

        # If a plan has worse reformatting & worse cost than another plan, we don't need to
        # consider it further.
        undominated_plans = Dict()
        for (plan_class_1, plan_1) in new_plans
            cost_1 = plan_1[2]
            output_compat_1 = plan_class_1[2]
            reformat_set_1 = plan_class_1[3]
            is_dominated = false
            for (plan_class_2, plan_2) in plans_by_set[plan_class_1[1]]
                cost_2 = plan_2[2]
                output_compat_2 = plan_class_2[2]
                reformat_set_2 = plan_class_2[3]
                if cost_1 > cost_2 && output_compat_1 >= output_compat_2 && reformat_set_2 ⊆ reformat_set_1
                    is_dominated = true
                    break
                end
            end
            if !is_dominated
                undominated_plans[plan_class_1] = plan_1
            end
        end
        optimal_plans = undominated_plans
    end

    # The cost of transposing the output as a second step if we choose to
    output_size = estimate_nnz(output_stats)
    transpose_cost = output_size * RandomWriteCost
    # The cost of writing to the output depending on whether the writes are sequential
    num_flops = get_prefix_iterations(all_vars, input_stats)

    min_cost = Inf
    best_prefix = nothing
    for (plan_class, plan) in optimal_plans
        output_compat = plan_class[2]
        rf_set = plan_class[3]
        cur_cost = plan[2]
        if output_compat == FULL_PREFIX
            cur_cost += output_size * SeqWriteCost
        elseif output_compat == SET_PREFIX
            cur_cost += output_size * SeqWriteCost + transpose_cost
        else
            cur_cost += num_flops * RandomWriteCost
        end
        for i in rf_set
            cur_cost += reformat_costs[i]
        end
        if cur_cost < min_cost
            min_cost = cur_cost
            best_prefix = plan[1]
        end
    end
    return best_prefix
end
