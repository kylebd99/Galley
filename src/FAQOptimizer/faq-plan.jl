# This file defines an FAQ sub-problem which we will optimize using specialized methods.
# These problems take the following form:
# ∑_V φ_1(V_1)⋅...⋅φ_k(V_k) where V_i ⊆ V
# The goal is to decompose the problem into a series of sub-expressions
# ∑_V' φ_i(V_i)⋅... (∑_V'' φ_j(V_j)⋅.... )
# These are evaluated by materializing the innermost expressions first and working up to the
# outer expression. Additionally, we may include a projection of a factor onto a subset of
# its variables, called an indicator factor. In this case, we refer to it as ψ_i, and it is
# either 0 or 1 depending on whether there exists a non-zero completion of the variables in φ_i.

@auto_hash_equals mutable struct Factor
    input::LogicalPlanNode
    active_indices::Set{IndexExpr}
    all_indices::Set{IndexExpr}
    is_indicator::Bool
    stats::TensorStats
    id::Int
end

@auto_hash_equals mutable struct FAQInstance
    mult_op::Function
    sum_op::Function
    output_indices::Set{IndexExpr}
    input_indices::Set{IndexExpr}
    factors::Set{Factor}
    output_index_order::Vector{IndexExpr}
    function FAQInstance(
        mult_op::Function,
        sum_op::Function,
        output_indices::Set{IndexExpr},
        input_indices::Set{IndexExpr},
        factors::Set{Factor})
        return new(mult_op, sum_op, output_indices, input_indices, factors, collect(output_indices))
    end

    function FAQInstance(
        mult_op::Function,
        sum_op::Function,
        output_indices::Set{IndexExpr},
        input_indices::Set{IndexExpr},
        factors::Set{Factor},
        output_index_order::Union{Nothing, Vector{IndexExpr}})
        return new(mult_op, sum_op, output_indices, input_indices, factors, output_index_order)
    end

end

@auto_hash_equals mutable struct Bag
    edge_covers::Set{Factor}
    covered_indices::Set{IndexExpr}
    parent_indices::Set{IndexExpr}
    child_bags::Set{Bag}
    stats::TensorStats
    id::Int

    function Bag(mult_op,
                    sum_op,
                    edge_covers::Set{Factor},
                    covered_indices::Set{IndexExpr},
                    parent_indices::Set{IndexExpr},
                    child_bags::Set{Bag},
                    id::Int)
        input_stats::Vector{TensorStats} = []
        for f in edge_covers
            push!(input_stats, f.stats)
        end
        for b in child_bags
            push!(input_stats, b.stats)
        end
        return new(edge_covers, covered_indices, parent_indices, child_bags, get_bag_stats(mult_op, sum_op, input_stats, parent_indices), id)
    end

    function Bag()
        new(Set(), Set(), Set(), Set(), Set(), TensorStats(), 0)
    end
end

@auto_hash_equals mutable struct HyperTreeDecomposition
    mult_op::Function
    sum_op::Function
    output_indices::Set{IndexExpr}
    root_bag::Bag
    output_index_order::Union{Nothing, Vector{IndexExpr}}
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
        result_node = Aggregate(sum_op, indices_to_aggregate, result_node)
    end
    return result_node
end

function decomposition_to_logical_plan(htd::HyperTreeDecomposition)
    expr = _recursive_bag_to_plan_node(htd.root_bag, htd.mult_op, htd.sum_op)
    if !isnothing(htd.output_index_order)
        expr = Reorder(expr, htd.output_index_order)
    end
    return expr
end
