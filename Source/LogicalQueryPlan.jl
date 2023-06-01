# This file defines the logical query plan (LQP) language. 
# Each LQP is a tree of expressions where interior nodes refer to 
# function calls and leaf nodes refer to constants or input tensors.

abstract type LogicalPlanNode end

mutable struct TensorStats
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
    op::Any
    indices::Vector{String}
    input::Any
    stats::Any
    parent::Any
end
ReduceDim(op, indices, input) = ReduceDim(op, indices, input, nothing, nothing)

mutable struct MapJoin <: LogicalPlanNode
    op::Any
    left_input::Any
    right_input::Any
    stats::Any
    parent::Any
end
MapJoin(op, left_input, right_input) = MapJoin(op, left_input, right_input, nothing, nothing)

mutable struct Reorder <: LogicalPlanNode
    input::Any
    index_order::Any
    stats::Any
    parent::Any
end
Reorder(input, index_order) = Reorder(input, index_order, nothing, nothing)

mutable struct InputTensor <: LogicalPlanNode
    tensor_id::String
    fiber::Finch.Fiber
    stats::TensorStats
    parent::Any
end

InputTensor(tensor_id::String, fiber::Finch.Fiber, stats::TensorStats) = InputTensor(tensor_id, fiber, stats, nothing)

function InputTensor(tensor_id::String, indices::Vector{String}, fiber::Fiber) 
    return InputTensor(tensor_id, fiber, TensorStats(indices, fiber))
end

Base.show(io::IO, input::InputTensor) = print(io, "InputTensor($(input.tensor_id), $(input.stats))") 

mutable struct Scalar <: LogicalPlanNode
    value::Any
    stats::TensorStats
    parent::Any
end
Scalar(value) = Scalar(value, nothing)
Scalar(value, stats::TensorStats) = Scalar(value, stats, nothing)
