# This file defines an FAQ sub-problem which we will optimize using specialized methods.
# These problems take the following form:
# ∑_V φ_1(V_1)⋅...⋅φ_k(V_k) where V_i ⊆ V
# The goal is to decompose the problem into a series of sub-expressions
# ∑_V' φ_i(V_i)⋅... (∑_V'' φ_j(V_j)⋅.... )
# These are evaluated by materializing the innermost expressions first and working up to the
# outer expression. Additionally, we may include a projection of a factor onto a subset of
# its variables, called an indicator factor. In this case, we refer to it as ψ_i, and it is
# either 0 or 1 depending on whether there exists a non-zero completion of the variables in φ_i.

struct Factor
    input::LogicalPlanNode
    active_indices::Set{IndexExpr}
    all_indices::Set{IndexExpr}
    is_indicator::Bool
    stats::TensorStats
end

struct FAQInstance
    mult_op::Function
    sum_op::Function
    output_indices::Set{IndexExpr}
    input_indices::Set{IndexExpr}
    factors::Vector{Factor}
end

mutable struct Bag
    edge_covers::Vector{Factor}
    covered_indices::Set{IndexExpr}
    parent_indices::Set{IndexExpr}
    child_bags::Vector{Bag}

    function Bag(edge_covers::Vector{Factor},
                    covered_indices::Set{IndexExpr},
                    parent_indices::Set{IndexExpr},
                    child_bags::Vector{Bag})
        return new(edge_covers, covered_indices, parent_indices, child_bags)
    end

    function Bag()
        new(nothing, [], Set(), Set(), [])
    end
end

mutable struct HyperTreeDecomposition
    mult_op::Function
    sum_op::Function
    output_indices::Set{IndexExpr}
    root_bag::Bag
end

function _factor_to_plan_node(f::Factor)
    if f.is_indicator
        throw("indicator factors are unimplemented")
    else
        return f.input
    end
end

function _recursive_bag_to_plan_node(b::Bag, mult_op::Function, sum_op::Function)
    factor_plan_nodes = LogicalPlanNode[]
    for f in b.edge_covers
        push!(factor_plan_nodes, _factor_to_plan_node(f))
    end
    for c in b.child_bags
        push!(factor_plan_nodes, _recursive_bag_to_plan_node(c, mult_op, sum_op))
    end
    result_node = nothing
    if length(factor_plan_nodes) >= 2
        result_node = MapJoin(mult_op, factor_plan_nodes[1], factor_plan_nodes[2])
        for i in 3:length(factor_plan_nodes)
            result_node = MapJoin(mult_op, result_node, factor_plan_nodes[i])
        end
    else
        result_node = factor_plan_nodes[1]
    end

    indices_to_aggregate = setdiff(b.covered_indices, b.parent_indices)

    if length(indices_to_aggregate) > 0
        result_node = Aggregate(sum_op, collect(indices_to_aggregate), result_node)
    end
    return result_node
end

function decomposition_to_logical_plan(htd::HyperTreeDecomposition)
    return _recursive_bag_to_plan_node(htd.root_bag, htd.mult_op, htd.sum_op)
end
