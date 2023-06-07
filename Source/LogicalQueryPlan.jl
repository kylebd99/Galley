using Finch
using AutoHashEquals
# This file defines the logical query plan (LQP) language. 
# Each LQP is a tree of expressions where interior nodes refer to 
# function calls and leaf nodes refer to constants or input tensors.

abstract type LogicalPlanNode end

@auto_hash_equals mutable struct TensorStats
    indices::Vector{String}
    dim_size::Dict{String, Int}
    cardinality::Float64
    default_value::Any
    index_order::Any
end
TensorStats(indices, dim_size, cardinality, default_value) = TensorStats(indices, dim_size, cardinality, default_value, nothing)
function TensorStats(indices::Vector{String}, fiber::Fiber)
    shape_tuple = size(fiber)
    dim_size = Dict()
    for i in 1:length(indices)
        dim_size[indices[i]] = shape_tuple[i]
    end
    cardinality = countstored(fiber)
    default_value = default(fiber)
    return TensorStats(indices, dim_size, cardinality, default_value, nothing)
end


mutable struct ReduceDim <: LogicalPlanNode
    head::Any
    args::Vector{Any}
    metadata::@NamedTuple{stats::Any, parent::Any}
end
ReduceDim(op, indices, input) = ReduceDim(op, [indices, input], (stats=nothing, parent=nothing))

mutable struct MapJoin <: LogicalPlanNode
    head::Any
    args::Vector{Any}
    metadata::@NamedTuple{stats::Any, parent::Any}
end
MapJoin(op, left_input, right_input) = MapJoin(op, [left_input, right_input], (stats=nothing, parent=nothing))

mutable struct Reorder <: LogicalPlanNode
    head::Any
    args::Vector{Any}
    metadata::@NamedTuple{stats::Any, parent::Any}
end
Reorder(input, index_order) = Reorder(nothing, [input, index_order], (stats=nothing, parent=nothing))

mutable struct InputTensor <: LogicalPlanNode
    head::Any
    args::Vector{Any}
    metadata::@NamedTuple{stats::Any, parent::Any}
end

InputTensor(tensor_id::String, fiber::Finch.Fiber, stats::TensorStats) = InputTensor(nothing, [tensor_id, fiber], (stats=stats, parent=nothing))

function InputTensor(tensor_id::String, indices::Vector{String}, fiber::Fiber) 
    return InputTensor(tensor_id, fiber, TensorStats(indices, fiber))
end

Base.show(io::IO, input::InputTensor) = print(io, "InputTensor($(input.tensor_id), $(input.stats))") 

mutable struct Scalar <: LogicalPlanNode
    head::Any
    args::Vector{Any}
    metadata::@NamedTuple{stats::Any, parent::Any}
end
Scalar(value) = Scalar(value, nothing)
Scalar(value, stats) = Scalar(nothing, [value], (stats=stats, parent=nothing))
