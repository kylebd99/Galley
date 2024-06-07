# This file defines how stats are produced for a logical expression based on its children.

# We begin by defining the necessary interface for a statistics object.
function merge_tensor_stats_join(op, all_stats::Vararg{TensorStats})
    throw(error("merge_tensor_stats_join not implemented for: ", typeof(all_stats[1])))
end

function merge_tensor_stats_union(op,  all_stats::Vararg{TensorStats})
    throw(error("merge_tensor_stats_union not implemented for: ", typeof(all_stats[1])))
end

function reduce_tensor_stats(op, reduce_indices::Set{IndexExpr}, stats::TensorStats)
    throw(error("reduce_tensor_stats not implemented for: ", typeof(stats)))
end

function transpose_tensor_stats(index_order::Vector{IndexExpr}, stats::TensorStats)
    throw(error("transpose_tensor_stats not implemented for: ", typeof(stats)))
end

# We now define a set of functions for manipulating the TensorDefs that will be shared
# across all statistics types
function merge_tensor_def_join(op, all_defs::Vararg{TensorDef})
    new_default_value = op([def.default_value for def in all_defs]...)
    new_index_set = union([def.index_set for def in all_defs]...)
    new_dim_sizes = Dict()
    for index in new_index_set
        for def in all_defs
            if index in def.index_set
                new_dim_sizes[index] = def.dim_sizes[index]
            end
        end
    end
    return TensorDef(new_index_set, new_dim_sizes, new_default_value, nothing, nothing, nothing)
end


function merge_tensor_def_union(op, all_defs::Vararg{TensorDef})
    new_default_value = op([def.default_value for def in all_defs]...)
    new_index_set = union([def.index_set for def in all_defs]...)
    new_dim_sizes = Dict()
    for index in new_index_set
        for def in all_defs
            if index in def.index_set
                new_dim_sizes[index] = def.dim_sizes[index]
            end
        end
    end
    return TensorDef(new_index_set, new_dim_sizes, new_default_value, nothing, nothing, nothing)
end

function reduce_tensor_def(op, reduce_indices::Set{IndexExpr}, def::TensorDef)
    op = op isa PlanNode ? op.val : op
    new_default_value = nothing
    if isidentity(op, def.default_value)
        new_default_value = def.default_value
    elseif op == +
        new_default_value = def.default_value * prod([def.dim_sizes[x] for x in reduce_indices])
    elseif op == *
        new_default_value = def.default_value ^ prod([def.dim_sizes[x] for x in reduce_indices])
    elseif isidempotent(op)
        new_default_value = op(def.default_value, def.default_value)
    else
        # This is going to be VERY SLOW. Should raise a warning about reductions over non-identity default values.
        # Depending on the semantics of reductions, we might be able to do this faster.
        println("Warning: A reduction can take place over a tensor whose default value is not the reduction operator's identity. \\
                         This can result in a large slowdown as the new default is calculated.")
        new_default_value = op([def.default_value for _ in prod([def.dim_sizes[x] for x in reduce_indices])]...)
    end

    new_index_set = setdiff(def.index_set, reduce_indices)
    new_dim_sizes = Dict()
    for index in new_index_set
        new_dim_sizes[index] = def.dim_sizes[index]
    end
    return TensorDef(new_index_set, new_dim_sizes, new_default_value, nothing, nothing, nothing)
end

# This function determines whether a binary operation is union-like or join-like and creates
# new statistics objects accordingly.
function merge_tensor_stats(op, all_stats::Vararg{ST}) where ST <: TensorStats
    join_like_args = []
    union_like_args = []
    for stats in all_stats
        if isannihilator(op, get_default_value(stats))
            push!(join_like_args, stats)
        else
            push!(union_like_args, stats)
        end
    end

    if length(union_like_args) == 0
        return merge_tensor_stats_join(op, all_stats...)
    elseif length(join_like_args) == 0
        return merge_tensor_stats_union(op, all_stats...)
    else
        join_stats = merge_tensor_stats_join(op, join_like_args...)
        return merge_tensor_stats_union(op, join_stats, union_like_args...)
    end
end

function merge_tensor_stats(op::PlanNode, all_stats::Vararg{ST}) where ST <:TensorStats
    return merge_tensor_stats(op.val, all_stats...)
end

function reduce_tensor_stats(op, reduce_indices::Union{Vector{PlanNode}, Set{PlanNode}}, stats::ST) where ST <:TensorStats
    return reduce_tensor_stats(op, Set{IndexExpr}([idx.name for idx in reduce_indices]), stats)
end

function transpose_tensor_def(index_order::Vector{IndexExpr}, def::TensorDef)
    return reindex_def(index_order, def)
end


################# NaiveStats Propagation ##################################################
 # We do everything in log for numerical stability
function merge_tensor_stats_join(op, all_stats::Vararg{NaiveStats})
    new_def = merge_tensor_def_join(op, [get_def(stats) for stats in all_stats]...)
    new_dim_space_size = sum([log2(get_dim_size(new_def, idx)) for idx in new_def.index_set])
    prob_non_default = sum([log2(stats.cardinality) - sum([log2(get_dim_size(stats, idx)) for idx in get_index_set(stats)]) for stats in all_stats])
    new_cardinality = 2^(prob_non_default + new_dim_space_size)
    return NaiveStats(new_def, new_cardinality)
end

function merge_tensor_stats_union(op, all_stats::Vararg{NaiveStats})
    new_def = merge_tensor_def_union(op, [get_def(stats) for stats in all_stats]...)
    new_dim_space_size = sum([log2(get_dim_size(new_def, idx)) for idx in new_def.index_set])
    prob_default = sum([log2(1 - 2^(log2(stats.cardinality) - sum([log2(get_dim_size(stats, idx)) for idx in get_index_set(stats)]))) for stats in all_stats])
    new_cardinality = 2^(log2(1 - 2^prob_default) + new_dim_space_size)
    return NaiveStats(new_def, new_cardinality)
end

function reduce_tensor_stats(op, reduce_indices::Set{IndexExpr}, stats::NaiveStats)
    if length(reduce_indices) == 0
        return deepcopy(stats)
    end
    new_def = reduce_tensor_def(op, reduce_indices, get_def(stats))
    new_dim_space_size = sum([log2(get_dim_size(new_def, idx)) for idx in new_def.index_set])
    old_dim_space_size = sum([log2(get_dim_size(stats, idx)) for idx in get_index_set(stats)])
    prob_default_value = 1 - 2^(log2(stats.cardinality)-old_dim_space_size)
    prob_non_default_subspace = 1 - 2^(log2(prob_default_value) * 2^(old_dim_space_size-new_dim_space_size))
    new_cardinality = 2^(new_dim_space_size + log2(prob_non_default_subspace))
    return NaiveStats(new_def, new_cardinality)
end

function transpose_tensor_stats(index_order::Vector{IndexExpr}, stats::NaiveStats)
    stats = deepcopy(stats)
    stats.def = transpose_tensor_def(index_order, get_def(stats))
    return stats
end


################# DCStats Propagation ##################################################
function merge_tensor_stats_join(op, all_stats::Vararg{DCStats})
    new_def = merge_tensor_def_join(op, [get_def(stats) for stats in all_stats]...)
    new_dc_dict = Dict()
    for dc in ∪([stats.dcs for stats in all_stats]...)
        dc_key = get_dc_key(dc)
        current_dc = get(new_dc_dict, dc_key, Inf)
        if dc.d < current_dc
            new_dc_dict[dc_key] = dc.d
        end
    end
    new_stats = DCStats(new_def, Set{DC}(DC(key.X, key.Y, d) for (key, d) in new_dc_dict))
    return new_stats
end

function merge_tensor_stats_union(op, all_stats::Vararg{DCStats})
    new_def = merge_tensor_def_union(op, [get_def(stats) for stats in all_stats]...)

    dc_keys = Set()
    stats_dcs = []
    # We start by extending all arguments' dcs to the new dimensions and infer dcs as needed
    for stats in all_stats
        condense_stats!(stats, timeout=100)
        dcs = Dict()
        new_idxs = collect(setdiff(get_index_set(stats), get_index_set(new_def)))
        Z = Set(new_idxs)
        for dc in stats.dcs
            dcs[(X=dc.X, Y=dc.Y)] = dc.d
            push!(dc_keys, (X=dc.X, Y=dc.Y))

            Z_dimension_space_size = get_dim_space_size(new_def, Z)
            ext_dc_key = (X=dc.X, Y=∪(dc.Y, Z))
            dcs[ext_dc_key] = dc.d*Z_dimension_space_size
            push!(dc_keys, ext_dc_key)
        end
        push!(stats_dcs, dcs)
    end

    # We only keep DCs which can be inferred from all inputs. Otherwise, we might miss
    # important information which simply wasn't inferred
    new_dcs = Dict()
    for key in dc_keys
        if all([haskey(dcs, key) for dcs in stats_dcs])
            new_dcs[key] = sum([get(dcs, key, 0) for dcs in stats_dcs])
        end
    end
    return DCStats(new_def, Set{DC}(DC(key.X, key.Y, d) for (key, d) in new_dcs))
end

function reduce_tensor_stats(op, reduce_indices::Set{IndexExpr}, stats::DCStats)
    if length(reduce_indices) == 0
        return deepcopy(stats)
    end
    new_def = reduce_tensor_def(op, reduce_indices, get_def(stats))
    new_dcs = deepcopy(stats.dcs)
    new_stats = DCStats(new_def, new_dcs)
    return new_stats
end

function transpose_tensor_stats(index_order::Vector{IndexExpr}, stats::DCStats)
    stats = deepcopy(stats)
    stats.def = transpose_tensor_def(index_order, get_def(stats))
    return stats
end
