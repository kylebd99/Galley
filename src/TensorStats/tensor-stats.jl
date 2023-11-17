@auto_hash_equals mutable struct TensorStats
    index_set::Set{IndexExpr}
    dim_size::Dict{IndexExpr, Int}
    cardinality::Float64
    default_value::Any
    index_order::Union{Nothing, Vector{IndexExpr}}
end

TensorStats(index_set, dim_size, cardinality, default_value) = TensorStats(index_set, dim_size, cardinality, default_value, nothing)
function TensorStats(indices::Vector{IndexExpr}, fiber::Fiber)
    shape_tuple = size(fiber)
    dim_size = Dict()
    for i in 1:length(indices)
        dim_size[indices[i]] = shape_tuple[i]
    end
    cardinality = countstored(fiber)
    default_value = default(fiber)
    return TensorStats(Set{IndexExpr}(indices), dim_size, cardinality, default_value, indices)
end

function TensorStats(x::Number)
    stats = TensorStats()
    stats.default_value = x
    return stats
end

TensorStats() = TensorStats(Set{IndexExpr}(), Dict(), -1, nothing, IndexExpr[])
