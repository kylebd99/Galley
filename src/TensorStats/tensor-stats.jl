
# A subset of the allowed level formats provided by the Finch API
@enum LevelFormat t_sparse_list = 1 t_dense = 2 t_hash = 3

# This struct holds the high-level definition of a tensor. This information should be
# agnostic to the statistics used for cardinality estimation.
@auto_hash_equals mutable struct TensorDef
    index_set::Set{IndexExpr}
    dim_sizes::Dict{IndexExpr, Int}
    default_value::Any
    level_formats::Union{Nothing, Vector{LevelFormat}}
    index_order::Union{Nothing, Vector{IndexExpr}}
end
TensorDef(default) = TensorDef(Set(), Dict(), default, nothing, nothing)

function level_to_enum(lvl)
    if typeof(lvl) <: SparseListLevel
        return t_sparse_list
    elseif typeof(lvl) <: SparseHashLevel
        return t_hash
    elseif typeof(lvl) <: DenseLevel
        return t_dense
    else
        throw(Base.error("Level Not Recognized"))
    end
end

function TensorDef(indices::Vector{IndexExpr}, tensor::Tensor)
    shape_tuple = size(tensor)
    dim_size = Dict()
    level_formats = LevelFormat[]
    current_lvl = tensor.lvl
    for i in 1:length(indices)
        dim_size[indices[i]] = shape_tuple[i]
        push!(level_formats, level_to_enum(current_lvl))
        current_lvl = current_lvl.lvl
    end
    default_value = Finch.default(tensor)
    return TensorDef(Set{IndexExpr}(indices), dim_size, default_value, level_formats, indices)
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

    return TensorDef(new_index_set, new_dim_sizes, def.default_value, def.level_formats, indices)
end

get_dim_sizes(def::TensorDef) = def.dim_sizes
get_dim_size(def::TensorDef, idx::IndexExpr) = def.dim_sizes[idx]
get_index_set(def::TensorDef) = def.index_set
get_index_order(def::TensorDef) = def.index_order
get_default_value(def::TensorDef) = def.default_value
get_index_format(def::TensorDef, idx::IndexExpr) = def.level_formats[findfirst(x->x==idx, def.index_order)]

function get_dim_space_size(def::TensorDef, indices::Set{IndexExpr})
    dim_space_size = 1
    for idx in indices
        dim_space_size *= def.dim_sizes[idx]
    end
    return dim_space_size
end

abstract type TensorStats end

get_dim_space_size(stat::TensorStats, indices::Set{IndexExpr}) = get_dim_space_size(get_def(stat), indices)
get_dim_sizes(stat::TensorStats) = get_dim_sizes(get_def(stat))
get_dim_size(stat::TensorStats, idx::IndexExpr) = get_dim_size(get_def(stat), idx)
get_index_set(stat::TensorStats) = get_index_set(get_def(stat))
get_index_order(stat::TensorStats) = get_index_order(get_def(stat))
get_default_value(stat::TensorStats) = get_default_value(get_def(stat))
get_index_format(stat::TensorStats, idx::IndexExpr) = get_index_format(get_def(stat), idx)


#################  NaiveStats Definition ###################################################

@auto_hash_equals mutable struct NaiveStats <: TensorStats
    def::TensorDef
    cardinality::Float64
end

get_def(stat::NaiveStats) = stat.def
estimate_nnz(stat::NaiveStats) = stat.cardinality
condense_stats(stat::NaiveStats) = stat

NaiveStats(default) = NaiveStats(TensorDef(default), 1)
NaiveStats(index_set, dim_sizes, cardinality, default_value) = NaiveStats(TensorDef(index_set, dim_sizes, default_value, nothing), cardinality)

function NaiveStats(indices::Vector{IndexExpr}, tensor::Tensor)
    def = TensorDef(indices, tensor)
    cardinality = countstored(tensor)
    return NaiveStats(def, cardinality)
end

function NaiveStats(x::Number)
    def = TensorDef(Set{IndexExpr}(), Dict{IndexExpr, Int}(), x, nothing, nothing)
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

function get_dc_key(dc::DegreeConstraint)
    return (X=dc.X, Y=dc.Y)
end

@auto_hash_equals mutable struct DCStats <: TensorStats
    def::TensorDef
    dcs::Set{DC}
end

DCStats(default) = DCStats(TensorDef(default), Set())
get_def(stat::DCStats) = stat.def

DCKey = NamedTuple{(:X, :Y), Tuple{Set{IndexExpr}, Set{IndexExpr}}}

# When we're only attempting to infer for nnz estimation, we only need to consider
# left dcs which have X = {}.
function _infer_dcs(dcs::Set{DC}; timeout=100000, cheap=false)
    all_dcs = Dict{DCKey, Float64}()
    for dc in dcs
        all_dcs[(X = dc.X, Y = dc.Y)] = dc.d
    end
    prev_new_dcs = deepcopy(all_dcs)
    time = 1
    finished = false
    while time < timeout && !finished
        new_dcs = Dict{DCKey, Float64}()
        function infer_dc(l, ld, r, rd)
            if l.Y ⊇ r.X
                new_key = (X = l.X, Y = setdiff(∪(l.Y, r.Y), l.X))
                new_degree = ld*rd
                if get(all_dcs, new_key, Inf) > new_degree &&
                        get(new_dcs, new_key, Inf) > new_degree
                    new_dcs[new_key] = new_degree
                end
            end
            time +=1
        end

        for (l, ld) in all_dcs
            cheap && length(l.X) > 0 && continue
            for (r, rd) in prev_new_dcs
                infer_dc(l, ld, r, rd)
                time > timeout && break
            end
            time > timeout && break
        end

        for (l, ld) in prev_new_dcs
            cheap && length(l.X) > 0 && continue
            for (r, rd) in all_dcs
                infer_dc(l, ld, r, rd)
                time > timeout && break
            end
            time > timeout && break
        end

        prev_new_dcs = new_dcs
        for (dc_key, dc) in new_dcs
            all_dcs[dc_key] = dc
        end
        if length(prev_new_dcs) == 0
            finished = true
        end
    end
    final_dcs = Set{DC}()
    for (dc_key, dc) in all_dcs
        push!(final_dcs, DC(dc_key.X, dc_key.Y, dc))
    end
    return final_dcs
end

function condense_stats(stat::DCStats)
    current_indices = get_index_set(stat)
    inferred_dcs = _infer_dcs(stat.dcs; cheap=false)
    min_dcs = Dict()
    for dc in inferred_dcs
        valid = true
        for x in dc.X
            if !(x in current_indices)
                valid = false
                break
            end
        end
        valid == false && break
        new_Y = dc.Y ∩ current_indices
        min_dcs[(dc.X, new_Y)] = min(get(min_dcs, (dc.X, new_Y), Inf), dc.d)
    end

    end_dcs = Set{DC}()
    for (dc_key, d) in min_dcs
        push!(end_dcs, DC(dc_key[1], dc_key[2], d))
    end
    stat.dcs = end_dcs
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

    inferred_dcs = _infer_dcs(dcs; cheap=true)
    min_card = Inf
    for dc in inferred_dcs
        if isempty(dc.X) && dc.Y ⊇ indices
            min_card = min(min_card, dc.d)
        end
    end
    return min_card
end

DCStats() = DCStats(TensorDef(), Set())

function _calc_dc_from_structure(X::Set{IndexExpr}, Y::Set{IndexExpr}, indices::Vector{IndexExpr}, s::Tensor)
    Z = setdiff(indices, ∪(X,Y)) # Indices that we want to project out before counting
    XY_ordered = setdiff(indices, Z)
    if length(Z) > 0
        XY_tensor = one_off_reduce(max, indices, XY_ordered, s)
    else
        XY_tensor = s
    end

    if length(XY_ordered) == 0
        return XY_tensor[]
    end

    X_ordered = collect(X)
    x_counts = one_off_reduce(+, XY_ordered, X_ordered, XY_tensor)
    if length(X) == 0
        return x_counts[] # If X is empty, we don't need to do a second pass
    end
    dc = one_off_reduce(max, X_ordered, IndexExpr[], x_counts)

    return dc[] # `[]` used to retrieve the actual value of the Finch.Scalar type
end

function _structure_to_dcs(indices::Vector{IndexExpr}, s::Tensor)
    dcs = Set{DC}()
    # Calculate DCs for all combinations of X and Y
    for X in subsets(indices)
        X = Set(X)
        Y = Set(setdiff(indices, X))
        d = _calc_dc_from_structure(X, Y, indices, s)
        push!(dcs, DC(X,Y,d))

        isempty(Y) && continue # Don't need to calculate the projection size of an empty set
        d = _calc_dc_from_structure(Set{IndexExpr}(), Y, indices, s)
        push!(dcs, DC(Set{IndexExpr}(), Y, d))
    end
    return dcs
end

function DCStats(indices::Vector{IndexExpr}, tensor::Tensor)
    def = TensorDef(indices, tensor)
    sparsity_structure = get_sparsity_structure(tensor)
    dcs = _structure_to_dcs(indices, sparsity_structure)
    return DCStats(def, dcs)
end

function DCStats(x::Number)
    def = TensorDef(Set{IndexExpr}(), Dict{IndexExpr, Int}(), x, nothing, nothing)
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
