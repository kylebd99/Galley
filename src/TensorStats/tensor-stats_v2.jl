# This struct holds the high-level definition of a tensor. This information should be
# agnostic to the statistics used for cardinality estimation. Any information which may be
# `Nothing` is considered a part of the physical definition which may be undefined for logical
# intermediates but is required to be defined for the inputs to an executable query.
@auto_hash_equals mutable struct TensorDef
    index_set::Set{IndexExpr}
    dim_sizes::Dict{IndexExpr, UInt128}
    default_value
    level_formats::Union{Nothing, Vector{LevelFormat}}
    index_order::Union{Nothing, Vector{IndexExpr}}
    index_protocols::Union{Nothing, Vector{AccessProtocol}}
end
TensorDef(x::Number) = TensorDef(Set(), Dict(), x, nothing, nothing, nothing)

copy_def(def::TensorDef) = TensorDef(Set(x for x in def.index_set),
                                        Dict(x for x in def.dim_sizes),
                                        def.default_value,
                                isnothing(def.level_formats) ? nothing : [x for x in def.level_formats],
                                isnothing(def.index_order) ? nothing : [x for x in def.index_order],
                                isnothing(def.index_protocols) ? nothing : [x for x in def.index_protocols])

function level_to_enum(lvl)
    if typeof(lvl) <: SparseListLevel
        return t_sparse_list
    elseif typeof(lvl) <: SparseDictLevel
        return t_hash
    elseif typeof(lvl) <: SparseByteMap
        return t_bytemap
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

function relabel_index!(def::TensorDef, i::IndexExpr, j::IndexExpr)
    if i == j || i ∉ def.index_set
        return
    end
    delete!(def.index_set, i)
    push!(def.index_set, j)
    def.dim_sizes[j] = def.dim_sizes[i]
    delete!(def.dim_sizes, i)
    if !isnothing(def.index_order)
        for k in eachindex(def.index_order)
            if def.index_order[k] == i
                def.index_order[k] = j
            end
        end
    end
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
    dim_space_size::UInt128 = 1
    for idx in indices
        dim_space_size *= def.dim_sizes[idx]
    end
    if dim_space_size == 0  || dim_space_size > typemax(Int)
        return UInt128(2)^63
    end
    return dim_space_size
end

abstract type TensorStats end

get_dim_space_size(stat::TensorStats, indices) = get_dim_space_size(get_def(stat), indices)
get_dim_sizes(stat::TensorStats) = get_dim_sizes(get_def(stat))
get_dim_size(stat::TensorStats, idx::IndexExpr) = get_dim_size(get_def(stat), idx)
get_index_set(stat::TensorStats) = get_index_set(get_def(stat))
get_index_order(stat::TensorStats) = get_index_order(get_def(stat))
get_default_value(stat::TensorStats) = get_default_value(get_def(stat))
get_index_format(stat::TensorStats, idx::IndexExpr) = get_index_format(get_def(stat), idx)
get_index_formats(stat::TensorStats) = get_index_formats(get_def(stat))
get_index_protocol(stat::TensorStats, idx::IndexExpr) = get_index_protocol(get_def(stat), idx)
get_index_protocols(stat::TensorStats) = get_index_protocols(get_def(stat))
copy_stats(stat::Nothing) = nothing
#################  NaiveStats Definition ###################################################

@auto_hash_equals mutable struct NaiveStats <: TensorStats
    def::TensorDef
    cardinality::Float64
end

get_def(stat::NaiveStats) = stat.def
estimate_nnz(stat::NaiveStats; indices = get_index_set(stat)) = stat.cardinality
condense_stats!(::NaiveStats; timeout=100000, cheap=true) = nothing
function fix_cardinality!(stat::NaiveStats, card)
    stat.cardinality = card
end
copy_stats(stat::NaiveStats) = NaiveStats(copy_def(stat.def), stat.cardinality)

NaiveStats(index_set, dim_sizes, cardinality, default_value) = NaiveStats(TensorDef(index_set, dim_sizes, default_value, nothing), cardinality)

function NaiveStats(tensor::Tensor, indices::Vector{IndexExpr})
    def = TensorDef(tensor, indices)
    cardinality = countstored(tensor)
    return NaiveStats(def, cardinality)
end

function NaiveStats(x::Number)
    def = TensorDef(Set{IndexExpr}(), Dict{IndexExpr, Int}(), x, nothing, nothing, nothing)
    return NaiveStats(def, 1)
end

function reindex_stats(stat::NaiveStats, indices::Vector{IndexExpr})
    return NaiveStats(reindex_def(indices, stat.def), stat.cardinality)
end

function relabel_index!(stats::NaiveStats, i::IndexExpr, j::IndexExpr)
    relabel_index!(stats.def, i, j)
end

#################  DCStats Definition ######################################################

struct DegreeConstraint
    X::SmallBitSet
    Y::SmallBitSet
    d::UInt128
end
DC = DegreeConstraint

function get_dc_key(dc::DegreeConstraint)
    return (X=dc.X, Y=dc.Y)
end

@auto_hash_equals mutable struct DCStats <: TensorStats
    def::TensorDef
    idx_2_int::Dict{IndexExpr, Int}
    int_2_idx::Dict{Int, IndexExpr}
    dcs::Set{DC}
end


copy_stats(stat::DCStats) = DCStats(copy_def(stat.def), copy(stat.idx_2_int), copy(stat.int_2_idx),  Set{DC}(dc for dc in stat.dcs))

DCStats(x::Number) = DCStats(TensorDef(x::Number), Dict(), Dict(), Set())
get_def(stat::DCStats) = stat.def

get_index_bitset(stat::DCStats) = SmallBitSet(stat.idx_2_int[x] for x in get_index_set(stat))

function fix_cardinality!(stat::DCStats, card)
    had_dc = false
    new_dcs = Set{DC}()
    for dc in stat.dcs
        if length(dc.X) == 0 && dc.Y == get_index_bitset(stat)
            push!(new_dcs, DC(SmallBitSet(), get_index_bitset(stat), min(card, dc.d)))
            had_dc = true
        else
            push!(new_dcs, dc)
        end
    end
    if !had_dc
        push!(new_dcs, DC(SmallBitSet(), get_index_bitset(stat), card))
    end
    stat.dcs = new_dcs
end

DCKey = NamedTuple{(:X, :Y), Tuple{SmallBitSet, SmallBitSet}}

function infer_dc(l, ld, r, rd, all_dcs, new_dcs)
    if l.Y ⊇ r.X
        new_key = (X = l.X, Y = setdiff!(∪(l.Y, r.Y), l.X))
        new_degree = ld*rd
        if get(all_dcs, new_key, Inf) > new_degree &&
                get(new_dcs, new_key, Inf) > new_degree
            new_dcs[new_key] = new_degree
        end
    end
end

# When we're only attempting to infer for nnz estimation, we only need to consider
# left dcs which have X = {}.
function _infer_dcs(dcs::Set{DC}; timeout=Inf, strength=0)
    all_dcs = Dict{DCKey, UInt128}()
    for dc in dcs
        all_dcs[(X = dc.X, Y = dc.Y)] = dc.d
    end
    prev_new_dcs = all_dcs
    time = 1
    finished = false
    max_dc_size = 0
    while time < timeout && !finished
        if strength <= 0
            max_dc_size = maximum([length(x.Y) for x in keys(prev_new_dcs)], init=0)
        end
        new_dcs = Dict{DCKey, UInt128}()

        for (l, ld) in all_dcs
            strength <= 1 && length(l.X) > 0 && continue
            for (r, rd) in prev_new_dcs
                strength <= 0 && length(r.Y) + length(l.Y) < max_dc_size && continue
                infer_dc(l, ld, r, rd, all_dcs, new_dcs)
                time +=1
                time > timeout && break
            end
            time > timeout && break
        end

        for (l, ld) in prev_new_dcs
            strength <= 1 && length(l.X) > 0 && continue
            for (r, rd) in all_dcs
                strength <= 0 && length(r.Y) + length(l.Y) < max_dc_size && continue
                infer_dc(l, ld, r, rd, all_dcs, new_dcs)
                time +=1
                time > timeout && break
            end
            time > timeout && break
        end

        prev_new_dcs = Dict()
        for (dc_key, dc) in new_dcs
            strength <= 0 && length(dc_key.Y) < max_dc_size && continue
            all_dcs[dc_key] = dc
            prev_new_dcs[dc_key] = dc
        end
        if length(prev_new_dcs) == 0
            finished = true
        end
    end
    final_dcs = Set{DC}()
    for (dc_key, dc) in all_dcs
        push!(final_dcs, DC(Set(dc_key.X), Set(dc_key.Y), dc))
    end
    return final_dcs
end

function condense_stats!(stat::DCStats; timeout=100000, cheap=false)
    current_indices_bitset = get_index_bitset(stat)
    inferred_dcs = nothing
    if !cheap
        inferred_dcs = _infer_dcs(stat.dcs; timeout=timeout, strength=2)
    else
        inferred_dcs = _infer_dcs(stat.dcs; timeout=min(timeout, 10000), strength=1)
        inferred_dcs = _infer_dcs(inferred_dcs; timeout=max(timeout - 10000, 0), strength=0)
    end
    min_dcs = Dict{DCKey, Float64}()
    for dc in inferred_dcs
        valid = true
        for x in dc.X
            if !(x in current_indices_bitset)
                valid = false
                break
            end
        end
        valid == false && continue
        new_Y = dc.Y ∩ current_indices_bitset
        y_dim_size = get_dim_space_size(stat.def, bitset_to_idxs(stat, new_Y))
        min_dcs[(X=dc.X, Y=new_Y)] = min(get(min_dcs, (X=dc.X, Y=new_Y), Inf), dc.d, y_dim_size)
    end

    end_dcs = Set{DC}()
    for (dc_key, d) in min_dcs
        push!(end_dcs, DC(dc_key.X, dc_key.Y, d))
    end
    stat.dcs = end_dcs
    return nothing
end

idxs_to_bitset(stat::DCStats, indices) = idxs_to_bitset(stat.idx_2_int, indices)
idxs_to_bitset(idx_2_int::Dict{IndexExpr, Int}, indices) = SmallBitSet(idx_2_int[idx] for idx in indices)
bitset_to_idxs(stat::DCStats, bitset) = bitset_to_idxs(stat.int_2_idx, bitset)
bitset_to_idxs(int_2_idx::Dict{Int, IndexExpr}, bitset) = Set{IndexExpr}(int_2_idx[idx] for idx in bitset)

function estimate_nnz(stat::DCStats; indices = get_index_set(stat))
    if length(indices) == 0
        return 1
    end
    indices_bitset = idxs_to_bitset(stat, indices)
    current_weights = Dict{SmallBitSet, UInt128}(SmallBitSet()=>1)
    frontier = Set{SmallBitSet}([SmallBitSet()])
    finished = false
    while !finished
        new_frontier = Set{SmallBitSet}()
        finished = true
        for x in frontier
            weight = current_weights[x]
            for dc in stat.dcs
                if x ⊇ dc.X
                    y = ∪(x, dc.Y)
                    if get(current_weights, y, Inf) >  weight * dc.d
                        # We need to be careful about overflow here. Turns out UInts overflow as 0 >:(
                        current_weights[y] = ((weight > (2^62)) || (dc.d > (2^62))) ? UInt128(2)^64 : (weight * dc.d)
                        finished = false
                        push!(new_frontier, y)
                    end
                end
            end
        end
        frontier = new_frontier
    end
    min_weight = Inf
    for (x, weight) in current_weights
        if x ⊇ indices_bitset
            min_weight = min(min_weight, weight)
        end
    end
    return min_weight
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


function _vector_structure_to_dcs(indices::Vector{Int}, s::Tensor)
    d_i = Scalar(0)
    @finch begin
        for i=_
            d_i[] += s[i]
        end
    end
    return Set{DC}([DC(SmallBitSet(), SmallBitSet(indices), d_i[])])
end

function _matrix_structure_to_dcs(indices::Vector{Int}, s::Tensor)
    X = Tensor(Dense(Element(0)))
    Y = Tensor(Dense(Element(0)))
    d_i = Scalar(0)
    d_j = Scalar(0)
    d_i_j = Scalar(0)
    d_j_i = Scalar(0)
    d_ij = Scalar(0)
    @finch begin
        X .= 0
        Y .= 0
        for i =_
            for j =_
                X[i] += s[j, i]
                Y[j] += s[j, i]
            end
        end
        d_i .= 0
        d_i_j .= 0
        d_ij .= 0
        for i=_
            d_i[] += X[i] > 0
            d_i_j[] <<max>>= X[i]
            d_ij[] += X[i]
        end
        d_j .= 0
        d_j_i .= 0
        for j=_
            d_j[] += Y[j] > 0
            d_j_i[] <<max>>= Y[j]
        end
    end
    i = indices[2]
    j = indices[1]
    return Set{DC}([DC(SmallBitSet(), SmallBitSet([i]), d_i[]),
                    DC(SmallBitSet(), SmallBitSet([j]), d_j[]),
                    DC(SmallBitSet([i]), SmallBitSet([j]), d_i_j[]),
                    DC(SmallBitSet([j]), SmallBitSet([i]), d_j_i[]),
                    DC(SmallBitSet(), SmallBitSet([i,j]), d_ij[]),
                    ])
end

function _structure_to_dcs(int_2_idx, indices::Vector{Int}, s::Tensor)
    if length(indices) == 1
        return _vector_structure_to_dcs(indices, s)
    elseif length(indices) == 2
        return _matrix_structure_to_dcs(indices, s)
    end
    dcs = Set{DC}()
    # Calculate DCs for all combinations of X and Y
    for X in subsets(indices)
        X = SmallBitSet(X)
        Y = SmallBitSet(setdiff(indices, X))
        isempty(Y) && continue # Anything to the empty set has degree 1
        d = _calc_dc_from_structure(bitset_to_idxs(int_2_idx, X), bitset_to_idxs(int_2_idx, Y), [int_2_idx[i] for i in indices], s)
        push!(dcs, DC(X,Y,d))

        d = _calc_dc_from_structure(Set{IndexExpr}(), bitset_to_idxs(int_2_idx, Y), [int_2_idx[i] for i in indices], s)
        push!(dcs, DC(SmallBitSet(), Y, d))
    end
    return dcs
end

function dense_dcs(def, int_2_idx, indices::Vector{Int})
    dcs = Set()
    for X in subsets(indices)
        Y = setdiff(indices, X)
        for Z in subsets(Y)
            push!(dcs, DC(SmallBitSet(X), SmallBitSet(Z), get_dim_space_size(def, bitset_to_idxs(int_2_idx, Z))))
        end
    end
    return dcs
end

function DCStats(tensor::Tensor, indices::Vector{IndexExpr})
    def = TensorDef(tensor, indices)
    idx_2_int = Dict{IndexExpr, Int}()
    int_2_idx = Dict{Int, IndexExpr}()
    for (i, idx) in enumerate(indices)
        idx_2_int[idx] = i
        int_2_idx[i] = idx
    end
    if all([f==t_dense for f in get_index_formats(def)])
        return DCStats(def, idx_2_int, int_2_idx, dense_dcs(def, int_2_idx, [i for i in range(1,length(indices))]))
    end
    sparsity_structure = pattern!(tensor)
    dcs = _structure_to_dcs(int_2_idx, [i for i in range(1,length(indices))], sparsity_structure)
    return DCStats(def, idx_2_int, int_2_idx, dcs)
end

function reindex_stats(stats::DCStats, indices::Vector{IndexExpr})
    new_def = reindex_def(indices, stats.def)
    rename_dict = Dict(get_index_order(stats)[i]=> indices[i] for i in eachindex(indices))
    new_idx_to_int = Dict{IndexExpr, Int}()
    for (idx, int) in stats.idx_2_int
        new_idx_to_int[rename_dict[idx]] = int
    end
    new_int_to_idx = Dict{Int, IndexExpr}()
    for (int, idx) in stats.int_2_idx
        new_int_to_idx[int] = rename_dict[idx]
    end
    return DCStats(new_def, new_idx_to_int, new_int_to_idx, copy(stats.dcs))
end

function relabel_index!(stats::DCStats, i::IndexExpr, j::IndexExpr)
    if i == j || i ∉ get_index_set(stats)
        return
    end
    relabel_index!(stats.def, i, j)
    stats.idx_2_int[j] = stats.idx_2_int[i]
    delete!(stats.idx_2_int, i)
    for (int, idx) in stats.int_2_idx
        if idx == i
            stats.int_2_idx[int] = j
            break
        end
    end
end
