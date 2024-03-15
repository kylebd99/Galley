
function select_leader_protocol(format::LevelFormat)
    if format == t_sparse_list
        return t_walk
    elseif format == t_dense
        return t_default
    elseif format == t_hash
        return t_walk
    end
end

function select_follower_protocol(format::LevelFormat)
    if format == t_sparse_list
        return t_default
    elseif format == t_dense
        return t_follow
    elseif format == t_hash
        return t_default
    end
end

function modify_protocols!(input_exprs)
    vars = union([i.input_indices for i in input_exprs]...)
    for var in vars
        relevant_inputs = [i for i in input_exprs if var ∈ i.input_indices]
        costs = []
        for input in relevant_inputs
            if get_index_format(input.stats, var) == t_dense
                push!(costs, get_dim_size(input.stats, var))
                continue
            end
            size_before_var = 1
            indices_before_var = []
            for index in input.input_indices
                index == var && break
                push!(indices_before_var, index)
            end
            if length(indices_before_var) > 0
                size_before_var = estimate_nnz(reduce_tensor_stats(+, setdiff(Set(input.input_indices), indices_before_var),  input.stats))
            end
            size_after_var = estimate_nnz(reduce_tensor_stats(+, setdiff(Set(input.input_indices), [indices_before_var..., var]),  input.stats))
            push!(costs, size_after_var/size_before_var)
        end
        min_cost = minimum(costs)
        needs_leader = length(relevant_inputs) > 1
        formats = [get_index_format(input.stats, var) for input in relevant_inputs]
        num_sparse_lists = sum([f == t_sparse_list for f in formats])
        use_gallop = false
        if num_sparse_lists > 1
            gallop_cost = minimum([costs[i] for i in eachindex(relevant_inputs) if formats[i] == t_sparse_list]) * RandomReadCost * 4
            walk_cost = maximum([costs[i] for i in eachindex(relevant_inputs) if formats[i] == t_sparse_list]) * SeqReadCost
            use_gallop = gallop_cost < walk_cost
        end
        for i in eachindex(relevant_inputs)
            input = relevant_inputs[i]
            var_index = findfirst(x->x==var, input.input_indices)
            is_leader = costs[i] == min_cost

            if formats[i] == t_sparse_list
                if use_gallop
                    input.input_protocols[var_index] = t_gallop
                else
                    input.input_protocols[var_index] = t_walk
                end
                needs_leader = false
                continue
            end

            if is_leader && needs_leader
                input.input_protocols[var_index] = select_leader_protocol(formats[i])
                needs_leader = false
            else
                input.input_protocols[var_index] = select_follower_protocol(formats[i])
            end
        end
    end
end
