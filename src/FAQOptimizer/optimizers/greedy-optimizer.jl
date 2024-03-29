# This optimizer is meant to run in time quadratic w.r.t. the FAQInstance. The goal is to
# aggressively pick one variable at a time, perform the associated joins & aggregate it
# out. Each time the variable whose associated join is smallest is selected.
function _get_index_cost(mult_op, sum_op, index::IndexExpr, inputs::Vector{Union{Factor{ST}, Bag{ST}}}, output_indices::Set{IndexExpr}) where ST
    edge_cover = Int[]
    edge_cover_stats = ST[]
    for i in eachindex(inputs)
        stats = inputs[i].stats
        if index in get_index_set(stats)
            push!(edge_cover, i)
            push!(edge_cover_stats, stats)
        end
    end
    covered_indices = union([get_index_set(stats) for stats in edge_cover_stats]...)

    # Add any edges which have been inadvertently covered by these additional indices.
    for i in eachindex(inputs)
        if i ∈ edge_cover
            continue
        end
        if get_index_set(inputs[i].stats) ⊆ covered_indices
            push!(edge_cover, i)
            push!(edge_cover_stats, inputs[i].stats)
        end
    end

    parent_indices = copy(output_indices)
    for idx in covered_indices
        for i in eachindex(inputs)
            i in edge_cover && continue
            if idx in get_index_set(inputs[i].stats)
                push!(parent_indices, idx)
                break
            end
        end
    end
    join_stat = merge_tensor_stats_join(mult_op, edge_cover_stats...)
    condense_stats!(join_stat)
    reduce_stat = reduce_tensor_stats(sum_op, setdiff(covered_indices, parent_indices), join_stat)
    cur_cost = estimate_nnz(reduce_stat)*AllocateCost + estimate_nnz(join_stat) * ComputeCost + sum([estimate_nnz(stats) for stats in edge_cover_stats]) * SeqReadCost
    return cur_cost, edge_cover, reduce_stat
end


function _get_cheapest_edge_cover(mult_op, sum_op, inputs::Vector{Union{Factor{ST}, Bag{ST}}}, output_indices::Set{IndexExpr}) where ST
    indices_to_aggregate = setdiff(union([get_index_set(input.stats) for input in inputs]...), output_indices)
    min_cost = Inf64
    cheapest_index = IndexExpr("")
    cheapest_edge_cover = Int[]
    cheapest_stat = nothing
    for index in indices_to_aggregate
        cur_cost, edge_cover, stat = _get_index_cost(mult_op, sum_op, index, inputs, output_indices)
        if cur_cost < min_cost
            min_cost = cur_cost
            cheapest_index = index
            cheapest_edge_cover = edge_cover
            cheapest_stat = stat
        end
    end
    if isinf(min_cost)
        println("MIN COST IS INFINITE!")
    end
    return cheapest_edge_cover, cheapest_stat
end

function greedy_decomposition(faq::FAQInstance)
    ST = typeof(first(faq.factors).stats)
    inputs = Vector{Union{Factor{ST}, Bag{ST}}}()
    mult_op = faq.mult_op
    sum_op = faq.sum_op
    output_indices = faq.output_indices
    output_index_order = faq.output_index_order
    factors = Set{Factor}(faq.factors)
    for factor in factors
        push!(inputs, factor)
    end
    all_indices = union([get_index_set(input.stats) for input in inputs]...)
    bag_counter = 0
    while all_indices != output_indices
        cheapest_edge_cover, cheapest_stat = _get_cheapest_edge_cover(mult_op, sum_op, inputs, output_indices)
        edge_cover = Set{Factor{ST}}()
        child_bags = Set{Bag{ST}}()
        covered_indices = Set{IndexExpr}()
        parent_indices = Set{IndexExpr}()
        for i in cheapest_edge_cover
            if inputs[i] isa Factor
                push!(edge_cover, inputs[i])
            else
                push!(child_bags, inputs[i])
            end
            for idx in get_index_set(inputs[i].stats)
                push!(covered_indices, idx)
            end
        end
        for idx in covered_indices
            for i in 1:length(inputs)
                if (!(i in cheapest_edge_cover) && idx in get_index_set(inputs[i].stats)) || idx in output_indices
                    push!(parent_indices, idx)
                    break
                end
            end
        end
        condense_stats!(cheapest_stat)
        new_bag = Bag(edge_cover,
                        covered_indices,
                        parent_indices,
                        child_bags,
                        cheapest_stat,
                        bag_counter)
        bag_counter += 1
        new_inputs = Union{Factor{ST}, Bag{ST}}[]
        for i in 1:length(inputs)
            if !(i in cheapest_edge_cover)
                push!(new_inputs, inputs[i])
            end
        end
        push!(new_inputs, new_bag)
        inputs = new_inputs
        all_indices = union([get_index_set(input.stats) for input in inputs]...)
    end
    child_bags = Set{Bag{ST}}()
    edge_covers = Set{Factor{ST}}()
    covered_indices = Set{IndexExpr}()
    for input in inputs
        if input isa Factor
            push!(edge_covers, input)
        else
            push!(child_bags, input)
        end
        for index in get_index_set(input.stats)
            push!(covered_indices, index)
        end
    end
    root_bag::Bag{ST} = Bag(mult_op, sum_op, edge_covers, covered_indices, output_indices, child_bags, bag_counter)
    htd = HyperTreeDecomposition(mult_op, sum_op, output_indices, root_bag, output_index_order)
    return htd
end
