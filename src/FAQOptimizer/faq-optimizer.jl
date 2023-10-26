

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
    next_vars::Vector{IndexExpr} = collect(E[v])
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
    while length(undecided_vars) > 0
        v = pop!(undecided_vars)
        component = _reachable_set(v, sep, E)
        undecided_vars = setdiff(undecided_vars, component)
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

function _recursive_hypertree_bag_decomp(factors::Vector{Factor},
                                        parent_vars::Set{IndexExpr},
                                        all_vars::Set{IndexExpr},
                                        max_width::Int)
    V = Set{IndexExpr}(union([f.all_indices for f in factors]...))
    for width in 1:max_width
        for edge_cover in subsets(factors, width)
            V1 = Set{IndexExpr}(union([f.all_indices for f in edge_cover]...))
            # This is important for maintaining the connectedness property
            if !(V ∩ parent_vars ⊆ V1)
                continue
            # This is a heuristic which eliminates a large swath of likely inefficient edge
            # covers
            elseif _check_for_cross_products(edge_cover)
                continue
            end
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
                return Bag(complete_edge_cover, V1, parent_vars, Bag[])
            end
            components = _get_components(V1, remaining_factors)
            child_trees = Bag[]
            invalid_cover = false
            for component in components
                component_edges = Factor[]
                for factor in remaining_factors
                    if length(intersect(factor.all_indices, component))  > 0
                        push!(component_edges, factor)
                    end
                end
                child_tree = _recursive_hypertree_bag_decomp(component_edges, V1, component, max_width)
                if isnothing(child_tree)
                    invalid_cover = true
                    break
                end
                push!(child_trees, child_tree)
            end
            invalid_cover && continue
            return Bag(complete_edge_cover, V1, parent_vars, child_trees)
        end
    end
    return nothing
end


function hypertree_decomposition(faq::FAQInstance)
    println("Beginning HTD")
    mult_op = faq.mult_op
    sum_op = faq.sum_op
    output_indices::Set{IndexExpr} = faq.output_indices
    variables::Set{IndexExpr} = faq.input_indices
    factors = faq.factors
    for max_width in 1:length(variables)
        bag = _recursive_hypertree_bag_decomp(factors, Set{IndexExpr}(), variables, max_width)
        if isnothing(bag)
            continue
        else
            println("Finished HTD")
            return HyperTreeDecomposition(mult_op, sum_op, output_indices, bag)
        end
    end
end


function faq_to_htd(faq::FAQInstance; faq_optimizer::FAQ_OPTIMIZERS = naive)
    if faq_optimizer == naive
        return naive_decomposition(faq)
    elseif faq_optimizer == hypertree
        return hypertree_decomposition(faq)
    else
        throw(ArgumentError(string(faq_optimizer) * " is not supported yet."))
    end
end
