# This file defines the logical query plan (LQP) language.
# Each LQP is a tree of expressions where interior nodes refer to
# function calls and leaf nodes refer to constants or input tensors.

# TODO: Change indices from strings to a struct so that we can capture interactions between
# and on them in the query plan, e.g. slicing and convolution.
@auto_hash_equals struct TensorStats
    indices::Vector{String}
    dim_size::Dict{String, Int}
    cardinality::Float64
    default_value::Any
    index_order::Any
end

TensorStats(indices, dim_size, cardinality, default_value) = TensorStats(indices, dim_size, cardinality, default_value, nothing)
function TensorStats(indices::Vector, fiber::Fiber, global_index_order)
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
mutable struct LogicalPlanNode
    head::Any
    args::Vector{Any}
    stats::Any
end

Aggregate(op, indices::Vector, input) = LogicalPlanNode(Aggregate, [op, indices, input], nothing)
Aggregate(op, index::String, input) = LogicalPlanNode(Aggregate, [op, [index], input], nothing)
Agg(op, indices, input) = Aggregate(op, indices, input)
MapJoin(op, left_input, right_input) = LogicalPlanNode(MapJoin, [op, left_input, right_input], nothing)
Reorder(input, index_order) = LogicalPlanNode(Reorder, [input, index_order], nothing)
InputTensor(fiber::Fiber)  = LogicalPlanNode(InputTensor, [[""], fiber], nothing)
Scalar(value, stats) = LogicalPlanNode(Scalar, [value], stats)
Scalar(value) = Scalar(value, nothing)
OutTensor() = LogicalPlanNode(OutTensor, [], nothing)
# This plan node generally occurs when an intermediate is re-used and new indices are applied.
RenameIndices(input, indices) = LogicalPlanNode(RenameIndices, [input, indices], nothing)


# This function allows users to natively apply custom binary operators.
# e.g. f(x,y) = (x + y)*2; declare_binary_operator(f)
function declare_binary_operator(f)
    @eval (::typeof($f))(l::LogicalPlanNode, r::LogicalPlanNode) = MapJoin($f, l, r)
    @eval (::typeof($f))(l::LogicalPlanNode, r) = MapJoin($f, l, r)
    @eval (::typeof($f))(l, r::LogicalPlanNode) = MapJoin($f, l, r)
end
declare_binary_operator(*)
declare_binary_operator(+)
declare_binary_operator(min)
declare_binary_operator(max)

∑(indices, input) = Aggregate(+, indices, input)
∏(indices, input) = Aggregate(*, indices, input)

# Indexing operator for tensors and expressions, e.g. A["i","j"]
function Base.getindex(input::LogicalPlanNode, indices...)
    if input.head == InputTensor
        return LogicalPlanNode(InputTensor, [collect(indices), input.args[2]], input.stats)
    else
        return LogicalPlanNode(RenameIndices, [input, collect(indices)], input.stats)
    end
end

# Indexed assignment for tensors, e.g. C["i", "j"] = TensorExpression
function Base.setindex!(output::LogicalPlanNode, input::LogicalPlanNode, indices...)
    output.head = Reorder
    output.args = [input, collect(indices)]
end

# In order to natively apply Metatheory to our query plans, we need to implement the
# major functions from TermInterface.
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

function logicalPlanToString(n::LogicalPlanNode, depth::Int64)
    output = ""
    if depth > 0
        output = "\n"
    end
    left_space = ""
    for _ in 1:depth
        left_space *= "   "
    end
    output *= left_space
    if n.head == InputTensor
        output *= "InputTensor("
    elseif n.head == Aggregate
        output *= "Aggregate("
    elseif n.head == MapJoin
        output *= "MapJoin("
    elseif n.head == Reorder
        output *= "Reorder("
    elseif n.head == Scalar
        output *= "Scalar("
    elseif n.head == RenameIndices
        output *= "RenameIndices("
    end

    prefix = ""
    for arg in n.args
        output *= prefix
        if arg isa LogicalPlanNode
            output *= logicalPlanToString(arg, depth + 1)
        elseif arg isa Fiber
            output *= "FIBER"
        else
            output *= string(arg)
        end
        prefix =","
    end
    output *= ")"
end

function Base.show(io::IO, input::LogicalPlanNode)
    print(io, logicalPlanToString(input, 0))
end
