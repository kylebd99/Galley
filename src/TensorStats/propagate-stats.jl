# This file defines how stats are produced for a logical expression based on its children.

# We begin by defining the necessary interface for a statistics object.
function merge_tensor_stats_join(op, all_stats::Vararg{TensorStats})
    throw(error("merge_tensor_stats_join not implemented for: ", typeof(all_stats[1])))
end

function merge_tensor_stats_union(op,  all_stats::Vararg{TensorStats})
    throw(error("merge_tensor_stats_union not implemented for: ", typeof(all_stats[1])))
end

function reduce_tensor_stats(op, reduce_indices::Set{IndexExpr}, stats::TensorStats)
    throw(error("merge_tensor_stats_union not implemented for: ", typeof(stats)))
end

function transpose_tensor_stats(index_order::Vector{IndexExpr}, stats::TensorStats)
    throw(error("merge_tensor_stats_union not implemented for: ", typeof(stats)))
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
    return TensorDef(new_index_set, new_dim_sizes, new_default_value, nothing, nothing)
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
    return TensorDef(new_index_set, new_dim_sizes, new_default_value, nothing, nothing)
end

function reduce_tensor_def(op, reduce_indices::Set{IndexExpr}, def::TensorDef)
    new_default_value = nothing
    if haskey(identity_dict, :($op)) && identity_dict[:($op)] == def.default_value
        new_default_value = def.default_value
    elseif op == +
        new_default_value = def.default_value * prod([def.dim_sizes[x] for x in reduce_indices])
    elseif op == *
        new_default_value = def.default_value ^ prod([def.dim_sizes[x] for x in reduce_indices])
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
    return TensorDef(new_index_set, new_dim_sizes, new_default_value, nothing, nothing)
end

# This function determines whether a binary operation is union-like or join-like and creates
# new statistics objects accordingly.
function merge_tensor_stats(op, all_stats::Vararg{ST}) where ST <: TensorStats
    if !haskey(annihilator_dict, :($op))
        return merge_tensor_stats_union(op, all_stats...)
    end

    annihilator_value = annihilator_dict[:($op)]
    if all([annihilator_value == get_default_value(stats) for stats in all_stats])
        return merge_tensor_stats_join(op, all_stats...)
    else
        return merge_tensor_stats_union(op, all_stats...)
    end
end


# This function takes in a logical plan node and annotates it as well as any children with
# a TensorStats object.
function _recursive_insert_stats!(n::LogicalPlanNode)
    if n.head == MapJoin
        map(_recursive_insert_stats!, n.args[2:end])
        n.stats = merge_tensor_stats(n.args[1], [arg.stats for arg in n.args[2:end]]...)
    elseif n.head == Aggregate
        _recursive_insert_stats!(n.args[3])
        n.stats = reduce_tensor_stats(n.args[1], n.args[2], n.args[3].stats)
    elseif n.head == RenameIndices
        _recursive_insert_stats!(n.args[1])
        n.stats = n.args[1].stats
    elseif n.head == Reorder
        _recursive_insert_stats!(n.args[1])
        n.stats = n.args[1].stats
    # If the stats of the input tensor has already been initialized, we leave it alone.
    # Otherwise, we use the stats type defined in the input.
    elseif n.head == InputTensor
        if n.stats isa Type
            n.stats = n.stats(n.args[1], n.args[2])
        else
            n.stats = n.stats
        end
    elseif n.head == Scalar
        TensorStats(n.args[1])
    end
    return n.stats
end

function transpose_tensor_def(index_order::Vector{IndexExpr}, def::TensorDef)
    return reindex_def(index_order, def)
end


################# NaiveStats Propagation ##################################################
function merge_tensor_stats_join(op, all_stats::Vararg{NaiveStats})
    new_def = merge_tensor_def_join(op, [get_def(stats) for stats in all_stats]...)
    new_dim_space_size = get_dim_space_size(new_def, new_def.index_set)
    prob_non_defaults = [stats.cardinality / get_dim_space_size(stats, get_index_set(stats)) for stats in all_stats]
    new_cardinality = prod(prob_non_defaults) * new_dim_space_size
    return NaiveStats(new_def, new_cardinality)
end

function merge_tensor_stats_union(op, all_stats::Vararg{NaiveStats})
    new_def = merge_tensor_def_union(op, [get_def(stats) for stats in all_stats]...)
    new_dim_space_size = get_dim_space_size(new_def, new_def.index_set)
    prob_defaults = [1 - stats.cardinality / get_dim_space_size(stats, get_index_set(stats)) for stats in all_stats]
    new_cardinality = (1 - prod(prob_defaults)) * new_dim_space_size
    return NaiveStats(new_def, new_cardinality)
end

function reduce_tensor_stats(op, reduce_indices::Set{IndexExpr}, stats::NaiveStats)
    new_def = reduce_tensor_def(op, reduce_indices, get_def(stats))
    new_dim_space_size = get_dim_space_size(stats, new_def.index_set)
    old_dim_space_size = get_dim_space_size(stats, get_index_set(stats))
    prob_default_value = 1 - stats.cardinality/old_dim_space_size
    prob_non_default_subspace = 1 - prob_default_value ^ (old_dim_space_size/new_dim_space_size)
    new_cardinality = new_dim_space_size * prob_non_default_subspace
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

    # We start by extending all arguments' dcs to the new dimensions
    new_dcs = Dict()
    for stats in all_stats
        new_idxs = setdiff(get_index_set(stats), get_index_set(new_def))
        for dc in stats.dcs
            for Z in subset(new_idxs)
                Z_dimension_space_size = get_dim_space_size(new_def, Z)
                dc_key = (X=dc.X, Y=∪(dc.Y, Z))
                current_dc = get(new_dc_dict, dc_key, 0)
                new_dcs[dc_key] = current_dc + dc.d*Z_dimension_space_size
            end
        end
    end
    return DCStats(new_def, Set{DC}(DC(key.X, key.Y, d) for (key, d) in new_dcs))
end

function reduce_tensor_stats(op, reduce_indices::Set{IndexExpr}, stats::DCStats)
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
