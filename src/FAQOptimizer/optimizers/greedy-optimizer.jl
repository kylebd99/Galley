# This optimizer is meant to run in time quadratic w.r.t. the FAQInstance. The goal is to
# aggressively pick one variable at a time, perform the associated joins & aggregate it
# out. Each time the variable whose associated join is smallest is selected.
function _get_index_cost(mult_op, sum_op, index::IndexExpr, inputs::Vector{Union{Factor, Bag}}, output_indices::Set{IndexExpr})
    edge_cover = Int[]
    edge_cover_stats = TensorStats[]
    for i in eachindex(inputs)
        stats = inputs[i].stats
        if index in get_index_set(stats)
            push!(edge_cover, i)
            push!(edge_cover_stats, stats)
        end
    end
    covered_indices = union([get_index_set(stats) for stats in edge_cover_stats]...)
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
    join_stat = edge_cover_stats[1]
    for i in 2:length(edge_cover_stats)
        join_stat = merge_tensor_stats_join(mult_op, join_stat, edge_cover_stats[i])
    end
    reduce_stat = reduce_tensor_stats(sum_op, setdiff(covered_indices, parent_indices), join_stat)
    cur_cost = estimate_nnz(reduce_stat) + estimate_nnz(join_stat) + sum([estimate_nnz(stats) for stats in edge_cover_stats])
    return cur_cost, edge_cover
end


function _get_cheapest_edge_cover(mult_op, sum_op, inputs::Vector{Union{Factor, Bag}}, output_indices::Set{IndexExpr})
    indices_to_aggregate = setdiff(union([get_index_set(input.stats) for input in inputs]...), output_indices)
    min_cost = Inf64
    cheapest_index = IndexExpr("")
    cheapest_edge_cover = Int[]
    for index in indices_to_aggregate
        cur_cost, edge_cover = _get_index_cost(mult_op, sum_op, index, inputs, output_indices)
        if cur_cost < min_cost
            min_cost = cur_cost
            cheapest_index = index
            cheapest_edge_cover = edge_cover
        end
    end
    return cheapest_edge_cover
end

function greedy_decomposition(faq::FAQInstance)
    inputs = Vector{Union{Factor, Bag}}()
    mult_op = faq.mult_op
    sum_op = faq.sum_op
    output_indices = faq.output_indices
    output_index_order = faq.output_index_order
    factors = Set{Factor}(faq.factors)
    for factor in factors
        push!(inputs, factor)
    end
    all_indices = union([get_index_set(input.stats) for input in inputs]...)
    while all_indices != output_indices
        cheapest_edge_cover = _get_cheapest_edge_cover(mult_op, sum_op, inputs, output_indices)
        edge_cover = Set{Factor}()
        child_bags = Set{Bag}()
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
        new_bag = Bag(mult_op,
                        sum_op,
                        edge_cover,
                        covered_indices,
                        parent_indices,
                        child_bags)
        new_inputs = Union{Factor, Bag}[]
        for i in 1:length(inputs)
            if !(i in cheapest_edge_cover)
                push!(new_inputs, inputs[i])
            end
        end
        push!(new_inputs, new_bag)
        inputs = new_inputs
        all_indices = union([get_index_set(input.stats) for input in inputs]...)
    end
    child_bags = Set{Bag}()
    edge_covers = Set{Factor}()
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
    root_bag::Bag = Bag(mult_op, sum_op, edge_covers, covered_indices, output_indices, child_bags)
    htd = HyperTreeDecomposition(mult_op, sum_op, output_indices, root_bag, output_index_order)
    return htd
end
