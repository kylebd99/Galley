

# This struct holds the high-level definition of a tensor. This information should be
# agnostic to the statistics used for cardinality estimation. Any information which may be
# `Nothing` is considered a part of the physical definition which may be undefined for logical
# intermediates but is required to be defined for the inputs to an executable query.
@auto_hash_equals mutable struct TensorDef
    index_set::Set{IndexExpr}
    dim_sizes::Dict{IndexExpr, Int}
    default_value::Any
    level_formats::Union{Nothing, Vector{LevelFormat}}
    index_order::Union{Nothing, Vector{IndexExpr}}
    index_protocols::Union{Nothing, Vector{AccessProtocol}}
end
TensorDef(default) = TensorDef(Set(), Dict(), default, nothing, nothing, nothing)

function level_to_enum(lvl)
    if typeof(lvl) <: SparseListLevel
        return t_sparse_list
    elseif typeof(lvl) <: SparseLevel
        return t_hash
    elseif typeof(lvl) <: DenseLevel
        return t_dense
    else
        throw(Base.error("Level Not Recognized"))
    end
end

function TensorDef(tensor::Tensor, indices::Vector{IndexExpr})
    shape_tuple = size(tensor)
    dim_size = Dict()
    level_formats = LevelFormat[]
    current_lvl = tensor.lvl
    for i in 1:length(indices)
        dim_size[indices[i]] = shape_tuple[i]
        push!(level_formats, level_to_enum(current_lvl))
        current_lvl = current_lvl.lvl
    end
    # Because levels are built outside-in, we need to reverse this.
    level_formats = reverse(level_formats)
    default_value = Finch.default(tensor)
    return TensorDef(Set{IndexExpr}(indices), dim_size, default_value, level_formats, indices, nothing)
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

    return TensorDef(new_index_set, new_dim_sizes, def.default_value, def.level_formats, indices, def.index_protocols)
end

get_dim_sizes(def::TensorDef) = def.dim_sizes
get_dim_size(def::TensorDef, idx::IndexExpr) = def.dim_sizes[idx]
get_index_set(def::TensorDef) = def.index_set
get_index_order(def::TensorDef) = def.index_order
get_default_value(def::TensorDef) = def.default_value
get_index_format(def::TensorDef, idx::IndexExpr) = def.level_formats[findfirst(x->x==idx, def.index_order)]
get_index_formats(def::TensorDef) = def.level_formats
get_index_protocol(def::TensorDef, idx::IndexExpr) = def.index_protocols[findfirst(x->x==idx, def.index_order)]
get_index_protocols(def::TensorDef) = def.index_protocols

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
get_index_formats(stat::TensorStats) = get_index_formats(get_def(stat))
get_index_protocol(stat::TensorStats, idx::IndexExpr) = get_index_protocol(get_def(stat), idx)
get_index_protocols(stat::TensorStats) = get_index_protocols(get_def(stat))

#################  NaiveStats Definition ###################################################

@auto_hash_equals mutable struct NaiveStats <: TensorStats
    def::TensorDef
    cardinality::Float64
end

get_def(stat::NaiveStats) = stat.def
estimate_nnz(stat::NaiveStats) = stat.cardinality
condense_stats!(::NaiveStats; timeout=100000, cheap=true) = nothing

NaiveStats(default) = NaiveStats(TensorDef(default), 1)
NaiveStats(index_set, dim_sizes, cardinality, default_value) = NaiveStats(TensorDef(index_set, dim_sizes, default_value, nothing), cardinality)

function NaiveStats(tensor::Tensor, indices::Vector{IndexExpr})
    def = TensorDef(tensor, indices)
    cardinality = countstored(tensor)
    return NaiveStats(def, cardinality)
end

function NaiveStats(x::Number)
    def = TensorDef(Set{IndexExpr}(), Dict{IndexExpr, Int}(), x, nothing, nothing)
    return NaiveStats(def, 1)
end

function reindex_stats(stat::NaiveStats, indices::Vector{IndexExpr})
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

DCKey = NamedTuple{(:X, :Y), Tuple{Vector{IndexExpr}, Vector{IndexExpr}}}

function union_then_diff(lY::Vector{IndexExpr}, rY::Vector{IndexExpr}, lX::Vector{IndexExpr})
    if length(lY) == 0
        return IndexExpr[]
    end
    if length(rY) == 0
        return copy(lY)
    end

    result = Vector{IndexExpr}(undef, length(lY) + length(rY))
    cur_idx = undef
    cur_out_idx = 1
    cur_l_pos = 1
    cur_r_pos = 1
    while cur_l_pos <= length(lY) || cur_r_pos <= length(rY)
        if cur_l_pos <= length(lY) && cur_r_pos <= length(rY)
            if rY[cur_r_pos] == lY[cur_l_pos]
                cur_idx = rY[cur_r_pos]
                cur_r_pos += 1
                cur_l_pos += 1
            elseif rY[cur_r_pos] < lY[cur_l_pos]
                cur_idx = rY[cur_r_pos]
                cur_r_pos += 1
            else
                cur_idx = lY[cur_l_pos]
                cur_l_pos += 1
            end
        elseif cur_l_pos <= length(lY)
            cur_idx = lY[cur_l_pos]
            cur_l_pos += 1
        elseif cur_r_pos <= length(rY)
            cur_idx = rY[cur_r_pos]
            cur_r_pos += 1
        end
        if !(cur_idx in lX)
            result[cur_out_idx] = cur_idx
            cur_out_idx += 1
        end
    end
    return result[1:cur_out_idx-1]
end

function infer_dc(l, ld, r, rd, all_dcs, new_dcs)
    if l.Y ⊇ r.X
        new_key = (X = l.X, Y = union_then_diff(l.Y, r.Y, l.X))
        new_degree = ld*rd
        if get(all_dcs, new_key, Inf) > new_degree &&
                get(new_dcs, new_key, Inf) > new_degree
            new_dcs[new_key] = new_degree
        end
    end
end

# When we're only attempting to infer for nnz estimation, we only need to consider
# left dcs which have X = {}.
function _infer_dcs(dcs::Set{DC}; timeout=Inf, cheap=false)
    all_dcs = Dict{DCKey, Float64}()
    for dc in dcs
        all_dcs[(X = sort!(collect(dc.X)), Y = sort!(collect(dc.Y)))] = dc.d
    end
    prev_new_dcs = all_dcs
    time = 1
    finished = false
    max_dc_size = maximum([length(x.Y) for x in keys(all_dcs)], init=0)
    while time < timeout && !finished
        new_dcs = Dict{DCKey, Float64}()

        for (l, ld) in all_dcs
            cheap && length(l.X) > 0 && continue
            for (r, rd) in prev_new_dcs
                cheap && length(r.Y) + length(l.Y) < max_dc_size && continue
                infer_dc(l, ld, r, rd, all_dcs, new_dcs)
                time +=1
                time > timeout && break
            end
            time > timeout && break
        end

        for (l, ld) in prev_new_dcs
            cheap && length(l.X) > 0 && continue
            for (r, rd) in all_dcs
                cheap && length(r.Y) + length(l.Y) < max_dc_size && continue
                infer_dc(l, ld, r, rd, all_dcs, new_dcs)
                time +=1
                time > timeout && break
            end
            time > timeout && break
        end

        if cheap
            max_dc_size = maximum([length(x.Y) for x in keys(new_dcs)], init=0)
        end
        prev_new_dcs = Dict()
        for (dc_key, dc) in new_dcs
            cheap && length(dc_key.Y) < max_dc_size && continue
            all_dcs[dc_key] = dc
            prev_new_dcs[dc_key] = dc
        end
        if length(prev_new_dcs) == 0
            finished = true
        end
    end
    time>timeout && println("Hit Timeout: $(time>timeout)")
    final_dcs = Set{DC}()
    for (dc_key, dc) in all_dcs
        push!(final_dcs, DC(Set(dc_key.X), Set(dc_key.Y), dc))
    end
    return final_dcs
end

function condense_stats!(stat::DCStats; timeout=Inf, cheap=true)
    current_indices = get_index_set(stat)
    inferred_dcs = _infer_dcs(stat.dcs; timeout=timeout, cheap=cheap)
    min_dcs = Dict()
    for dc in inferred_dcs
        valid = true
        for x in dc.X
            if !(x in current_indices)
                valid = false
                break
            end
        end
        valid == false && continue
        new_Y = dc.Y ∩ current_indices
        min_dcs[(dc.X, new_Y)] = min(get(min_dcs, (dc.X, new_Y), Inf), dc.d)
    end

    end_dcs = Set{DC}()
    for (dc_key, d) in min_dcs
        push!(end_dcs, DC(dc_key[1], dc_key[2], d))
    end
    stat.dcs = end_dcs
    return nothing
end


function estimate_nnz(stat::DCStats)
    indices = get_index_set(stat)
    if length(indices) == 0
        return 1
    end
    dcs = stat.dcs
    min_card = Inf
    for dc in dcs
        if isempty(dc.X) && dc.Y ⊇ indices
            min_card = min(min_card, dc.d)
        end
    end
    min_card < Inf && return min_card
    inferred_dcs = _infer_dcs(dcs; cheap=true)
    for dc in inferred_dcs
        if isempty(dc.X) && dc.Y ⊇ indices
            min_card = min(min_card, dc.d)
        end
    end
    if min_card == Inf
        println("ESTIMATED INF!")
    end
    return min_card
end

DCStats() = DCStats(TensorDef(), Set())

function _calc_dc_from_structure(X::Set{IndexExpr}, Y::Set{IndexExpr}, indices::Vector{IndexExpr}, s::Tensor)
    Z = [i for i in indices if i ∉ ∪(X,Y)] # Indices that we want to project out before counting
    XY_ordered = [i for i in indices if i ∉ Z]
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

function DCStats(tensor::Tensor, indices::Vector{IndexExpr})
    def = TensorDef(tensor, indices)
    sparsity_structure = get_sparsity_structure(tensor)
    dcs = _structure_to_dcs(indices, sparsity_structure)
    return DCStats(def, dcs)
end

function DCStats(x::Number)
    def = TensorDef(Set{IndexExpr}(), Dict{IndexExpr, Int}(), x, nothing, nothing)
    return DCStats(def, Set([DC(Set(), Set(), 1)]))
end

function reindex_stats(stat::DCStats, indices::Vector{IndexExpr})
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
