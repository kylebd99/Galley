
function select_leader_protocol(format::LevelFormat)
    if format == t_sparse_list
        return t_walk
    elseif format == t_dense
        return t_default
    elseif format == t_bytemap
        return t_walk
    elseif format == t_hash
        return t_walk
    end
end

function select_follower_protocol(format::LevelFormat)
    if format == t_sparse_list
        return t_default
    elseif format == t_dense
        return t_follow
    elseif format == t_bytemap
        return t_follow
    elseif format == t_hash
        return t_default
    end
end

function modify_protocols!(input_stats::Vector{ST}) where ST
    for input in input_stats
        get_def(input).index_protocols = [t_default for _ in get_index_order(input)]
    end

    vars = union([get_index_set(i) for i in input_stats]...)
    for var in vars
        relevant_inputs = [i for i in input_stats if var ∈ get_index_set(i)]
        costs = []
        for input in relevant_inputs
            if get_index_format(input, var) == t_dense
                push!(costs, get_dim_size(input, var))
                continue
            end
            size_before_var = 1
            indices_before_var = []
            for index in reverse(get_index_order(input))
                index == var && break
                push!(indices_before_var, index)
            end
            # The choice of `min` below is arbitrary because the actual agg_op doesn't affect
            # the nnz (barring things like prod reductions which might be a TODO).
            if length(indices_before_var) > 0
                size_before_var = estimate_nnz(reduce_tensor_stats(min, setdiff(get_index_set(input), indices_before_var),  input))
            end
            size_after_var = estimate_nnz(reduce_tensor_stats(min, setdiff(get_index_set(input), [indices_before_var..., var]),  input))
            push!(costs, max(1, size_after_var/size_before_var))
        end
        min_cost = minimum(costs)
        needs_leader = true
        formats = [get_index_format(input, var) for input in relevant_inputs]
        num_sparse_lists = sum([f == t_sparse_list for f in formats])
        use_gallop = false
        if num_sparse_lists > 1
            gallop_cost = minimum([costs[i] for i in eachindex(relevant_inputs) if formats[i] == t_sparse_list]) * RandomReadCost
            walk_cost = maximum([costs[i] for i in eachindex(relevant_inputs) if formats[i] == t_sparse_list]) * SeqReadCost
            use_gallop = gallop_cost < walk_cost
            # It seems as though Gallop is generally the correct choice, and it is
            # asymptotically better than walk. So, we just always set it to be conservative.
            use_gallop = true
            needs_leader = false
        end
        for i in eachindex(relevant_inputs)
            input = relevant_inputs[i]
            input_def = get_def(input)
            var_index = findall(x->x==var, get_index_order(input))
            is_leader = costs[i] == min_cost
            if formats[i] == t_sparse_list
                if use_gallop
                    input_def.index_protocols[var_index] .= t_gallop
                else
                    input_def.index_protocols[var_index] .= t_walk
                end
                needs_leader = false
                continue
            end

            if is_leader && needs_leader
                input_def.index_protocols[var_index] .= select_leader_protocol(formats[i])
                needs_leader = false
            else
                input_def.index_protocols[var_index] .= select_follower_protocol(formats[i])
            end
        end
    end
end
