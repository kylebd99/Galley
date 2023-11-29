# This struct holds the high-level definition of a tensor. This information should be
# agnostic to the statistics used for cardinality estimation.
@auto_hash_equals mutable struct TensorDef
    index_set::Set{IndexExpr}
    dim_sizes::Dict{IndexExpr, Int}
    default_value::Any
    index_order::Union{Nothing, Vector{IndexExpr}}
end

function TensorDef(indices::Vector{IndexExpr}, fiber::Fiber)
    shape_tuple = size(fiber)
    dim_size = Dict()
    for i in 1:length(indices)
        dim_size[indices[i]] = shape_tuple[i]
    end
    default_value = default(fiber)
    return TensorDef(Set{IndexExpr}(indices), dim_size, default_value, indices)
end

abstract type TensorStats end


@auto_hash_equals mutable struct NaiveStats <: TensorStats
    def::TensorDef
    cardinality::Float64
end

get_def(stat::NaiveStats) = stat.def
get_index_set(stat::NaiveStats) = stat.def.index_set
get_dim_sizes(stat::NaiveStats) = stat.def.dim_sizes
get_dim_size(stat::NaiveStats, idx::IndexExpr) = stat.def.dim_sizes[idx]
get_default_value(stat::NaiveStats) = stat.def.default_value
get_index_order(stat::NaiveStats) = stat.def.index_order
estimate_nnz(stat::NaiveStats) = stat.cardinality

NaiveStats() = NaiveStats(TensorDef(), 0)
NaiveStats(index_set, dim_size, cardinality, default_value) = NaiveStats(index_set, dim_size, cardinality, default_value, nothing)

function NaiveStats(indices::Vector{IndexExpr}, fiber::Fiber)
    def = TensorDef(indices, fiber)
    cardinality = countstored(fiber)
    return NaiveStats(def, cardinality)
end

function NaiveStats(x::Number)
    def = TensorDef(Set{IndexExpr}(), Dict{IndexExpr, Int}(), x, nothing)
    return NaiveStats(def, 1)
end
