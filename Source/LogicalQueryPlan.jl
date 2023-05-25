# This file defines the logical query plan (LQP) language. 
# Each LQP is a tree of expressions where interior nodes refer to 
# function calls and leaf nodes refer to constants or input tensors.

abstract type LogicalPlanNode end

mutable struct TensorStats
    indices::Set{String}
    dim_size::Dict{String, Int}
    cardinality::Float64
    default_value::Any
end

mutable struct ReduceDim <: LogicalPlanNode
    op::Any
    indices::Set{String}
    input::Any
    stats::Any
    parent::Any
end

mutable struct MapJoin <: LogicalPlanNode
    op::Any
    left_input::Any
    right_input::Any
    stats::Any
    parent::Any
end

mutable struct InputTensor <: LogicalPlanNode
    tensor_id::String
    index_order::Vector{String}
    stats::TensorStats
    parent::Any
end
InputTensor(tensor_id::String, index_order::Vector{String}, stats::TensorStats) = InputTensor(tensor_id, index_order, stats, nothing)

mutable struct Scalar <: LogicalPlanNode
    value::Any
    stats::TensorStats
    parent::Any
end
Scalar(value, stats::TensorStats) = Scalar(value, stats, nothing)



function e_graph_to_expr_tree(g::EGraph)
    return e_class_to_expr_node(g, g[g.root])
end

function e_class_to_expr_node(g::EGraph, e::EClass; verbose=0)
    n = e[1]
    stats = getdata(e, :TensorStatsAnalysis)
    children = []
    if n isa ENodeTerm
        for c in arguments(n)
            if g[c][1] isa ENodeTerm
                push!(children, e_class_to_expr_node(g, g[c]))
            elseif g[c][1].value isa InputTensor
                push!(children, g[c][1].value)
            elseif g[c][1].value isa Number
                push!(children, Scalar(g[c][1].value, getdata(g[c], :TensorStatsAnalysis)))            
            else
                push!(children, g[c][1].value)
            end
        end
    end
    nodeType = eval(operation(n))
    return nodeType(children..., stats, nothing)
end

function label_expr_parents!(parent, cur_node::LogicalPlanNode)
    cur_node.parent = parent
    if cur_node isa ReduceDim && cur_node.input isa LogicalPlanNode
            label_expr_parents!(cur_node, cur_node.input)
    elseif cur_node isa MapJoin
        if cur_node.left_input isa LogicalPlanNode
            label_expr_parents!(cur_node, cur_node.left_input)
        end
        if cur_node.right_input isa LogicalPlanNode
            label_expr_parents!(cur_node, cur_node.right_input)
        end
    end
    return cur_node
end