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
    default_value = Finch.default(fiber)
    return TensorDef(Set{IndexExpr}(indices), dim_size, default_value, indices)
end

function reindex_def(indices::Vector{IndexExpr}, def::TensorDef)
    @assert length(indices) == length(def.index_order)
    rename_dict = Dict()
    for i in eachindex(indices)
        rename_dict[def.index_order[i]] = indices[i]
    end
    new_index_set = Set{IndexExpr}()
    for idx in def.index_set
        push!(new_index_set, rename_dict[idx])
    end

    new_dim_sizes = Dict()
    for (idx, size) in def.dim_sizes
        new_dim_sizes[rename_dict[idx]] = size
    end

    return TensorDef(new_index_set, new_dim_sizes, def.default_value, indices)
end

abstract type TensorStats end


function get_dim_space_size(def::TensorDef, indices::Set{IndexExpr})
    dim_space_size = 1
    for idx in indices
        dim_space_size *= def.dim_sizes[idx]
    end
    return dim_space_size
end

function get_dim_space_size(stat::TensorStats, indices::Set{IndexExpr})
    return get_dim_space_size(get_def(stat), indices)
end

#################  NaiveStats Definition ###################################################

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
NaiveStats(index_set, dim_sizes, cardinality, default_value) = NaiveStats(TensorDef(index_set, dim_sizes, default_value, nothing), cardinality)

function NaiveStats(indices::Vector{IndexExpr}, fiber::Fiber)
    def = TensorDef(indices, fiber)
    cardinality = countstored(fiber)
    return NaiveStats(def, cardinality)
end

function NaiveStats(x::Number)
    def = TensorDef(Set{IndexExpr}(), Dict{IndexExpr, Int}(), x, nothing)
    return NaiveStats(def, 1)
end

function reindex_stats(indices::Vector{IndexExpr}, stat::NaiveStats)
    return NaiveStats(reindex_def(indices, stat.def), stat.cardinality)
end

#################  DCStats Definition ######################################################

struct DegreeConstraint
    X::Set{IndexExpr}
    Y::Set{IndexExpr}
    d::Float64
end
DC = DegreeConstraint

@auto_hash_equals mutable struct DCStats <: TensorStats
    def::TensorDef
    dcs::Set{DC}
end

get_def(stat::DCStats) = stat.def
get_index_set(stat::DCStats) = stat.def.index_set
get_dim_sizes(stat::DCStats) = stat.def.dim_sizes
get_dim_size(stat::DCStats, idx::IndexExpr) = stat.def.dim_sizes[idx]
get_default_value(stat::DCStats) = stat.def.default_value
get_index_order(stat::DCStats) = stat.def.index_order


function _infer_dcs(dcs::Set{DC}; timeout=100000)
    all_dcs = Dict()
    for dc in dcs
        all_dcs[(X = dc.X, Y = dc.Y)] = dc
    end
    prev_new_dcs = deepcopy(all_dcs)
    new_dcs = Dict()
    time = 1
    finished = false
    while time < timeout && !finished
        function infer_dc(l, r)
            if l.Y ⊇ r.X
                new_key = (X = l.X, Y = ∪(l.Y, r.Y))
                new_degree = l.d*r.d
                if get(all_dcs, new_key, DC(new_key.X, new_key.Y, Inf)).d > new_degree &&
                        get(new_dcs, new_key, DC(new_key.X, new_key.Y, Inf)).d > new_degree
                    new_dcs[new_key] = DC(new_key.X, new_key.Y, new_degree)
                end
            elseif r.Y ⊇ l.X
                new_key = (X = r.X, Y = ∪(r.Y, l.Y))
                new_degree = r.d*l.d
                if get(all_dcs, new_key, DC(new_key.X, new_key.Y, Inf)).d > new_degree &&
                    get(new_dcs, new_key, DC(new_key.X, new_key.Y, Inf)).d > new_degree
                    new_dcs[new_key] = DC(new_key.X, new_key.Y, new_degree)
                end
            end
            time +=1
        end

        for l in values(all_dcs)
            for r in values(prev_new_dcs)
                infer_dc(l,r)
                time > timeout && break
            end
            time > timeout && break
        end
        prev_new_dcs = new_dcs
        for dc in values(new_dcs)
            all_dcs[(X = dc.X, Y = dc.Y)] = dc
        end
        new_dcs = Dict()
        if length(prev_new_dcs) == 0
            finished = true
        end
    end
    final_dcs = Set{DC}()
    for dc in values(all_dcs)
        push!(final_dcs, dc)
    end
    return final_dcs
end

function estimate_nnz(stat::DCStats)
    indices = get_index_set(stat)
    dcs = stat.dcs
    min_card = Inf
    for dc in dcs
        if isempty(dc.X) && dc.Y ⊇ indices
            min_card = min(min_card, dc.d)
        end
    end
    min_card < Inf && return min_card

    inferred_dcs = _infer_dcs(dcs)
    min_card = Inf
    for dc in inferred_dcs
        if isempty(dc.X) && dc.Y ⊇ indices
            min_card = min(min_card, dc.d)
        end
    end
    return min_card
end

DCStats() = DCStats(TensorDef(), Set())

function _calc_dc_from_structure(X::Set{IndexExpr}, Y::Set{IndexExpr}, indices::Vector{IndexExpr}, s::Fiber)
    println("Calculating DC: ", string((X,Y)))
    println("Stored: ", countstored(s))
    Z = setdiff(indices, ∪(X,Y)) # Indices that we want to project out before counting
    XY_ordered = setdiff(indices, Z)
    if length(Z) > 0
        XY_fiber = one_off_reduce(max, indices, XY_ordered, s)
    else
        XY_fiber = s
    end

    if length(XY_ordered) == 0
        return XY_fiber[]
    end

    X_ordered = collect(X)
    x_counts = one_off_reduce(+, XY_ordered, X_ordered, XY_fiber)
    if length(X) == 0
        return x_counts[] # If X is empty, we don't need to do a second pass
    end
    dc = one_off_reduce(max, X_ordered, IndexExpr[], x_counts)
    return dc[] # `[]` used to retrieve the actual value of the Finch.Scalar type
end

function _structure_to_dcs(indices::Vector{IndexExpr}, s::Fiber)
    dcs = Set{DC}()
    # Calculate DCs for all combinations of X and Y
    for X in subsets(indices)
        X = Set(X)
        Y = Set(setdiff(indices, X))
        d = _calc_dc_from_structure(X, Y, indices, s)
        push!(dcs, DC(X,Y,d))

        isempty(X) && continue # Don't need to calculate the projection size of an empty set
        X == Set(indices) && continue

        X = Set(X)
        Y = Set(setdiff(indices, X))
        d = _calc_dc_from_structure(Set{IndexExpr}(), X, indices, s)
        push!(dcs, DC(Set{IndexExpr}(), X, d))
    end
    return dcs
end

function DCStats(indices::Vector{IndexExpr}, fiber::Fiber)
    def = TensorDef(indices, fiber)
    println(typeof(fiber))
    sparsity_structure = get_sparsity_structure(fiber)
    dcs = _structure_to_dcs(indices, sparsity_structure)
    return DCStats(def, dcs)
end

function DCStats(x::Number)
    def = TensorDef(Set{IndexExpr}(), Dict{IndexExpr, Int}(), x, nothing)
    return DCStats(def, Set([DC(Set(), Set(), 1)]))
end

function reindex_stats(indices::Vector{IndexExpr}, stat::DCStats)
    new_def = reindex_def(indices, stat.def)
    rename_dict = Dict(get_index_order(stat)[i]=> indices[i] for i in eachindex(indices))
    new_dcs = Set()
    for dc in stat.dcs
        new_X = Set(rename_dict[x] for x in dc.X)
        new_Y = Set(rename_dict[y] for y in dc.Y)
        push!(new_dcs, DC(new_X, new_Y, dc.d))
    end
    return DCStats(new_def, new_dcs)
end
