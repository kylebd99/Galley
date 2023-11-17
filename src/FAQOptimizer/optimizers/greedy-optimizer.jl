# This optimizer is meant to run in time quadratic w.r.t. the FAQInstance. The goal is to
# aggressively pick one variable at a time, perform the associated joins & aggregate it
# out. Each time the variable whose associated join is smallest is selected.
function _get_index_cost(mult_op, index::IndexExpr, inputs::Vector{Union{Factor, Bag}})
    edge_cover = Int[]
    edge_cover_stats = TensorStats[]
    for i in eachindex(inputs)
        stats = inputs[i].stats
        if index in stats.index_set
            push!(edge_cover, i)
            push!(edge_cover_stats, stats)
        end
    end
    covered_indices = union([stats.index_set for stats in edge_cover_stats]...)
    parent_indices = Set{IndexExpr}()
    for idx in covered_indices
        for input in inputs
            if idx in input.stats.index_set
                push!(parent_indices, idx)
                break
            end
        end
    end
    resulting_stat = edge_cover_stats[1]
    for i in 2:length(edge_cover_stats)
        resulting_stat = merge_tensor_stats_join(mult_op, resulting_stat, edge_cover_stats[i])
    end
    cur_cost = resulting_stat.cardinality + sum([stats.cardinality for stats in edge_cover_stats])
    return cur_cost, edge_cover
end


function _get_cheapest_edge_cover(mult_op, inputs::Vector{Union{Factor, Bag}})
    all_indices = union([input.stats.index_set for input in inputs]...)
    min_cost = Inf64
    cheapest_index = IndexExpr("")
    cheapest_edge_cover = Int[]
    for index in all_indices
        cur_cost, edge_cover = _get_index_cost(mult_op, index, inputs)
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
    factors = Set{Factor}(faq.factors)
    for factor in factors
        push!(inputs, factor)
    end
    while length(inputs) > 1
        cheapest_edge_cover = _get_cheapest_edge_cover(mult_op, inputs)
        edge_cover = Factor[]
        child_bags = Bag[]
        covered_indices = Set{IndexExpr}()
        parent_indices = Set{IndexExpr}()
        for i in cheapest_edge_cover
            if inputs[i] isa Factor
                push!(edge_cover, inputs[i])
            else
                push!(child_bags, inputs[i])
            end
            for idx in inputs[i].stats.index_set
                push!(covered_indices, idx)
            end
        end
        for idx in covered_indices
            for i in 1:length(inputs)
                if !(i in cheapest_edge_cover) && idx in inputs[i].stats.index_set
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
    end
    if inputs[1] isa Factor
        inputs[1] = Bag(mult_op, sum_op, [inputs[1]], Set{IndexExpr}(inputs[1].stats.index_set), Set{IndexExpr}(inputs[1].stats.index_set), Bag[])
    end
    root_bag::Bag = inputs[1]
    htd = HyperTreeDecomposition(mult_op, sum_op, output_indices, root_bag)
    return htd
end
