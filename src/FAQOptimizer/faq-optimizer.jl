

@enum FAQ_OPTIMIZERS naive hypertree

function naive_decomposition(faq::FAQInstance)
    mult_op = faq.mult_op
    sum_op = faq.sum_op
    output_indices = faq.output_indices
    factors = faq.factors
    bag::Bag = Bag(factors, faq.input_indices, faq.output_indices, Bag[])
    return HyperTreeDecomposition(mult_op, sum_op, output_indices, bag)
end


function _reachable_set(v::IndexExpr, sep::Set{IndexExpr}, E::Dict{IndexExpr, Set{IndexExpr}})
    visited_vars = Set{IndexExpr}([v]) ∪ sep
    next_vars::Vector{IndexExpr} = setdiff(collect(E[v]), visited_vars)
    while length(next_vars) > 0
        v2 = pop!(next_vars)
        push!(visited_vars, v2)
        for v3 in setdiff(E[v2], visited_vars)
            push!(next_vars, v3)
        end
    end
    return setdiff(visited_vars, sep)
end

function _get_components(sep::Set{IndexExpr}, factors::Vector{Factor})
    E::Dict{IndexExpr, Set{IndexExpr}} = _factors_to_adjacency_dict(factors)
    undecided_vars = collect(setdiff(Set(keys(E)), sep))
    components = []
#    println("BREAK")
#    println("E: ", E)
    while length(undecided_vars) > 0
        v = pop!(undecided_vars)
        component = _reachable_set(v, sep, E)
        undecided_vars = setdiff(undecided_vars, component)
#        println("V: ", v)
#        println("Sep: ", sep)
#        println("Component: ", component)
#        println("undecided_vars: ", undecided_vars)
        push!(components, component)
    end
    return components
end

function _factors_to_adjacency_dict(factors::Vector{Factor})
    adjacency_dict::Dict{IndexExpr, Set{IndexExpr}} = Dict()
    for factor in factors
        for v1 in factor.all_indices
            if !haskey(adjacency_dict, v1)
                adjacency_dict[v1] = Set()
            end
            for v2 in factor.all_indices
                push!(adjacency_dict[v1], v2)
            end
        end
    end
    return adjacency_dict
end

function _check_for_cross_products(factors::Vector{Factor})
    V = Set{IndexExpr}(union([f.all_indices for f in factors]...))
    connected_vars = _get_components(Set{IndexExpr}(), factors)[1]
    return V != connected_vars
end

function get_valid_subsets(factors::Set{Factor}, width, factor_graph::Dict{Factor, Vector{Factor}})
    factor_sets_by_width = Dict{Int, Set{Set{Factor}}}()
    prev_width_sets::Set{Set{Factor}} = Set{Set{Factor}}([Set([factor]) for factor in factors])
    factor_sets_by_width[1] = prev_width_sets
    for i in 2:width
        new_width_sets = Set{Set{Factor}}()
        for factor_set in prev_width_sets
            neighbors = ∪([factor_graph[factor] for factor in factor_set]...)
            for neighbor in neighbors
                new_set = factor_set ∪ Set([neighbor])
                push!(new_width_sets, new_set)
            end
        end
        prev_width_sets = new_width_sets
        factor_sets_by_width[i] = new_width_sets
    end
    return factor_sets_by_width
end

function _recursive_hypertree_bag_decomp(factors::Set{Factor},
                                        factor_sets_by_width::Dict{Int, Set{Set{Factor}}},
                                        parent_vars::Set{IndexExpr},
                                        max_width::Int,
                                        subtree_dict::Dict{Tuple{Set{Factor}, Set{IndexExpr}}, Any})
    haskey(subtree_dict, (factors, parent_vars)) && return subtree_dict[(factors, parent_vars)]
    V = Set{IndexExpr}(union([f.all_indices for f in factors]...))
    V1 = Set{IndexExpr}()
    for width in 1:max_width
        for edge_cover in factor_sets_by_width[width]
            factors_in_subgraph = true
            for factor in edge_cover
                if factor ∉ factors
                    factors_in_subgraph = false
                end
            end
            (!factors_in_subgraph) && continue

            # This is important for maintaining the connectedness property
            empty!(V1)
            for factor in edge_cover
                for idx in factor.all_indices
                    push!(V1, idx)
                end
            end
            not_connected = false
            for v in parent_vars
                if v ∈ V && v ∉ V1
                    not_connected = true
                    break
                end
            end
            not_connected && continue

            complete_edge_cover = Factor[]
            remaining_factors = Factor[]
            for factor in factors
                if factor.all_indices ⊆ V1
                    push!(complete_edge_cover, factor)
                else
                    push!(remaining_factors, factor)
                end
            end

            if length(remaining_factors) == 0
                bag = Bag(complete_edge_cover, V1, parent_vars, Bag[])
                subtree_dict[(factors, parent_vars)] = bag
                return bag
            end

            components = _get_components(V1, remaining_factors)
            child_trees = Bag[]
            invalid_cover = false
            for component in components
                component_edges = Set{Factor}()
                for factor in remaining_factors
                    if length(intersect(factor.all_indices, component))  > 0
                        push!(component_edges, factor)
                    end
                end
                child_tree = _recursive_hypertree_bag_decomp(component_edges, factor_sets_by_width, V1, max_width, subtree_dict)
                if isnothing(child_tree)
                    invalid_cover = true
                    break
                end
                push!(child_trees, child_tree)
            end
            invalid_cover && continue
            bag = Bag(complete_edge_cover, V1, parent_vars, child_trees)
            subtree_dict[(factors, parent_vars)] = bag
            return bag
        end
    end
    return nothing
end


function hypertree_decomposition(faq::FAQInstance)
    println("Beginning HTD")
    mult_op = faq.mult_op
    sum_op = faq.sum_op
    output_indices::Set{IndexExpr} = faq.output_indices
    factors = Set{Factor}(faq.factors)
    println("Making Factor Graph")
    factor_graph = Dict{Factor, Vector{Factor}}()
    for factor in factors
        neighbors = Factor[]
        for factor2 in factors
            if factor == factor2
                continue
            end
            if length(factor.all_indices ∩ factor2.all_indices) > 0
                push!(neighbors, factor2)
            end
        end
        factor_graph[factor] = neighbors
    end
    println("Done Making Factor Graph")
    start_time = time()
    subtree_dict = Dict{Tuple{Set{Factor}, Set{IndexExpr}}, Any}()
    for max_width in 1:length(factors)
        println("Length: ", time() - start_time)
        println("Max Width: ", max_width)
        println("Getting Subsets")
        factor_sets_by_width = get_valid_subsets(factors, max_width, factor_graph)
        println("Done Getting Subsets")
        bag = _recursive_hypertree_bag_decomp(factors, factor_sets_by_width, Set{IndexExpr}(), max_width, subtree_dict)
        if isnothing(bag)
            continue
        else
            println("Finished HTD")
            return HyperTreeDecomposition(mult_op, sum_op, output_indices, bag)
        end
    end
end

function make_join_factor(op, lfactor::Factor, rfactor::Factor)
    return Factor(MapJoin(op, lfactor.input, rfactor.input),
                    union(lfactor.active_indices,rfactor.active_indices),
                    union(lfactor.all_indices, rfactor.all_indices),
                    false,
                    merge_tensor_stats_join(op, lfactor.stats, rfactor.stats))
end

function merge_prefix_joins!(faq::FAQInstance)
    made_change = false
    finished = false
    prev_factors::Set{Factor} = Set(faq.factors)
    while !finished
        new_factors = Set{Factor}()
        handled_factors = Set{Factor}()
        for factor in prev_factors
            factor in handled_factors && continue
            for sub_factor in prev_factors
                sub_factor in handled_factors && continue
                factor == sub_factor && continue
                if factor.all_indices ⊇ sub_factor.all_indices
                    push!(new_factors, make_join_factor(faq.mult_op, factor, sub_factor))
                    push!(handled_factors, factor)
                    push!(handled_factors, sub_factor)
                    break
                end
            end
        end

        for factor in prev_factors
            if !(factor in handled_factors)
                push!(new_factors, factor)
            end
        end

        if prev_factors == new_factors
            finished = true
        else
            made_change = true
        end
        prev_factors = new_factors
    end
    faq.factors = collect(prev_factors)
    return made_change
end

function aggregate_dangling_vars!(faq::FAQInstance)
    made_change = false
    indices = copy(faq.input_indices)
    for idx in indices
        rel_factors = Factor[]
        for factor in faq.factors
            if idx in factor.all_indices
                push!(rel_factors, factor)
            end
        end
        if length(rel_factors) == 1 && !(idx in faq.output_indices)
            factor = rel_factors[1]
            factor.input = Aggregate(faq.sum_op, [idx], factor.input)
            delete!(factor.active_indices, idx)
            delete!(factor.all_indices, idx)
            delete!(faq.input_indices, idx)
            made_change = true
        end
    end
    return made_change
end

function prune_faq!(faq::FAQInstance)
    finished = false
    while !finished
        merged_joins = merge_prefix_joins!(faq)
        aggregated_indices = aggregate_dangling_vars!(faq)
        finished = !(merged_joins || aggregated_indices)
    end
end


function faq_to_htd(faq::FAQInstance; faq_optimizer::FAQ_OPTIMIZERS = naive)
    prune_faq!(faq)
    if faq_optimizer == naive
        return naive_decomposition(faq)
    elseif faq_optimizer == hypertree
        return hypertree_decomposition(faq)
    else
        throw(ArgumentError(string(faq_optimizer) * " is not supported yet."))
    end
end
