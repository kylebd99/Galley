using Finch
using AutoHashEquals
using TermInterface

# This file defines the logical query plan (LQP) language. 
# Each LQP is a tree of expressions where interior nodes refer to 
# function calls and leaf nodes refer to constants or input tensors.
@auto_hash_equals struct TensorStats
    indices::Vector{String}
    dim_size::Dict{String, Int}
    cardinality::Float64
    default_value::Any
    index_order::Any
end

TensorStats(indices, dim_size, cardinality, default_value) = TensorStats(indices, dim_size, cardinality, default_value, nothing)
function TensorStats(indices::Vector{String}, fiber::Fiber, global_index_order)
    shape_tuple = size(fiber)
    dim_size = Dict()
    for i in 1:length(indices)
        dim_size[indices[i]] = shape_tuple[i]
    end
    cardinality = countstored(fiber)
    default_value = default(fiber)
    return TensorStats(indices, dim_size, cardinality, default_value, global_index_order)
end


# Here, we define the internal expression type that we use to describe logical query plans.
# This is the expression type that we will optimize using Metatheory.jl, so it has to conform to 
# the TermInterface specs.
struct LogicalPlanNode
    head::Any
    args::Vector{Any}
    stats::Any
end

ReduceDim(op, indices, input) = LogicalPlanNode(ReduceDim, [op, indices, input], nothing)
MapJoin(op, left_input, right_input) = LogicalPlanNode(MapJoin, [op, left_input, right_input], nothing)
Reorder(input, index_order) = LogicalPlanNode(Reorder, [input, index_order], nothing)
InputTensor(tensor_id::String, indices::Vector{String}, fiber::Fiber)  = LogicalPlanNode(InputTensor, [tensor_id, indices, fiber], nothing)
Scalar(value, stats) = LogicalPlanNode(Scalar, [value], stats)
Scalar(value) = Scalar(value, nothing)

TermInterface.istree(::LogicalPlanNode) = true
TermInterface.operation(node::LogicalPlanNode) = node.head
TermInterface.arguments(node::LogicalPlanNode) = node.args
TermInterface.exprhead(::LogicalPlanNode) = :call
TermInterface.metadata(node::LogicalPlanNode) = node.stats
function TermInterface.similarterm(x::LogicalPlanNode, head, args; metadata = nothing, exprhead = :call)
    return LogicalPlanNode(head, args, x.stats)
end

function EGraphs.egraph_reconstruct_expression(::Type{LogicalPlanNode}, op, args; metadata = nothing, exprhead = nothing)
    return LogicalPlanNode(op, args, metadata)
end

function pprintLogicalPlan(io::IO, n::LogicalPlanNode, depth::Int64)
    print(io, "\n")
    left_space = ""
    for _ in 1:depth
        left_space *= "   "
    end
    print(io, left_space)
    if n.head == InputTensor
        print(io, "InputTensor(")
    elseif n.head == ReduceDim 
        print(io, "ReduceDim(")
    elseif n.head == MapJoin 
        print(io, "MapJoin(")
    elseif n.head == Reorder 
        print(io, "Reorder(")
    elseif n.head == Scalar
        print(io, "Scalar(")
    end
    prefix = ""
    for arg in n.args
        print(io, prefix)
        if arg isa LogicalPlanNode
            pprintLogicalPlan(io, arg, depth + 1)
        elseif arg isa Fiber
            print(io, "FIBER")
        else
            print(io, arg)
        end
        prefix =","
    end
    print(io, ")")
end

function Base.show(io::IO, input::LogicalPlanNode) 
    pprintLogicalPlan(io, input, 0)
end 
