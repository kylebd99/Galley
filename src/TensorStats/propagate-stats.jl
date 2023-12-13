# This file defines how stats are produced for a logical expression based on its children.

# We begin by defining the necessary interface for a statistics object.
function merge_tensor_stats_join(op, lstats::TensorStats, rstats::TensorStats)
    throw(error("merge_tensor_stats_join not implemented for: ", typeof(lstats), "  ", typeof(rstats)))
end

function merge_tensor_stats_union(op, lstats::TensorStats, rstats::TensorStats)
    throw(error("merge_tensor_stats_union not implemented for: ", typeof(lstats), "  ", typeof(rstats)))
end

function reduce_tensor_stats(op, reduce_indices::Set{IndexExpr}, stats::TensorStats)
    throw(error("merge_tensor_stats_union not implemented for: ", typeof(stats)))
end

# We now define a set of functions for manipulating the TensorDefs that will be shared
# across all statistics types
function merge_tensor_def_join(op, ldef::TensorDef, rdef::TensorDef)
    new_default_value = op(ldef.default_value, rdef.default_value)
    new_index_set = union(ldef.index_set, rdef.index_set)
    new_dim_sizes = Dict()
    for index in new_index_set
        if index in ldef.index_set && index in rdef.index_set
            # Here, we assume that indices can only be shared between tensors with the same
            # dimensions.
            @assert ldef.dim_sizes[index] == rdef.dim_sizes[index]
            new_dim_sizes[index] = ldef.dim_sizes[index]
        elseif index in rdef.index_set
            new_dim_sizes[index] = rdef.dim_sizes[index]
        else
            new_dim_sizes[index] = ldef.dim_sizes[index]
        end
    end
    return TensorDef(new_index_set, new_dim_sizes, new_default_value, nothing)
end


function merge_tensor_def_union(op, ldef::TensorDef, rdef::TensorDef)
    new_default_value = op(ldef.default_value, rdef.default_value)
    new_index_set = union(ldef.index_set, rdef.index_set)

    new_dim_sizes = Dict()
    for index in new_index_set
        if index in ldef.index_set && index in rdef.index_set
            # Here, we assume that indices can only be shared between tensors with the same
            # dimensions.
            @assert ldef.dim_sizes[index] == rdef.dim_sizes[index]
            new_dim_sizes[index] = ldef.dim_sizes[index]
        elseif index in rdef.index_set
            new_dim_sizes[index] = rdef.dim_sizes[index]
        else
            new_dim_sizes[index] = rdef.dim_sizes[index]
        end
    end
    return TensorDef(new_index_set, new_dim_sizes, new_default_value, nothing)
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
    return TensorDef(new_index_set, new_dim_sizes, new_default_value, nothing)
end

# This function determines whether a binary operation is union-like or join-like and creates
# new statistics objects accordingly.
function merge_tensor_stats(op, lstats::TensorStats, rstats::TensorStats)
    if length(get_index_set(lstats)) == 0
        new_stats = deepcopy(rstats)
        new_def = get_def(new_stats)
        new_def.default_value = op(get_default_value(lstats), get_default_value(rstats))
        return new_stats
    elseif length(get_index_set(rstats)) == 0
        new_stats = deepcopy(lstats)
        new_def = get_def(new_stats)
        new_def.default_value = op(get_default_value(lstats), get_default_value(rstats))
        return new_stats
    end

    if !haskey(annihilator_dict, :($op))
        return merge_tensor_stats_union(op, lstats, rstats)
    end

    annihilator_value = annihilator_dict[:($op)]
    if annihilator_value == get_default_value(lstats) && annihilator_value == get_default_value(rstats)
        return merge_tensor_stats_join(op, lstats, rstats)
    else
        return merge_tensor_stats_union(op, lstats, rstats)
    end
end


# This function takes in a logical plan node and annotates it as well as any children with
# a TensorStats object.
function _recursive_insert_stats!(n::LogicalPlanNode)
    if n.head == MapJoin
        _recursive_insert_stats!(n.args[2])
        _recursive_insert_stats!(n.args[3])
        n.stats = merge_tensor_stats(n.args[1], n.args[2].stats, n.args[3].stats)
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


################# NaiveStats Propagation ##################################################
function merge_tensor_stats_join(op, lstats::NaiveStats, rstats::NaiveStats)
    new_def = merge_tensor_def_join(op, get_def(lstats), get_def(rstats))
    new_dim_space_size = get_dim_space_size(new_def, new_def.index_set)
    l_dim_space_size = get_dim_space_size(lstats, get_index_set(lstats))
    r_dim_space_size = get_dim_space_size(rstats, get_index_set(rstats))
    l_prob_non_default = (lstats.cardinality/l_dim_space_size)
    r_prob_non_default = (rstats.cardinality/r_dim_space_size)
    new_cardinality = l_prob_non_default * r_prob_non_default * new_dim_space_size
    return NaiveStats(new_def, new_cardinality)
end

function merge_tensor_stats_union(op, lstats::NaiveStats, rstats::NaiveStats)
    new_def = merge_tensor_def_union(op, get_def(lstats), get_def(rstats))
    new_dim_space_size = get_dim_space_size(new_def, new_def.index_set)
    l_dim_space_size = get_dim_space_size(lstats, get_index_set(lstats))
    r_dim_space_size = get_dim_space_size(rstats, get_index_set(rstats))
    l_prob_default = (1 - lstats.cardinality/l_dim_space_size)
    r_prob_default = (1 - rstats.cardinality/r_dim_space_size)
    new_cardinality = (1 - l_prob_default * r_prob_default) * new_dim_space_size
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


################# DCStats Propagation ##################################################
function merge_tensor_stats_join(op, lstats::DCStats, rstats::DCStats)
    new_def = merge_tensor_def_join(op, get_def(lstats), get_def(rstats))
    new_dc_dict = Dict()
    for dc in ∪(rstats.dcs, lstats.dcs)
        dc_key = (X=dc.X, Y=dc.Y)
        current_dc = get(new_dc_dict, dc_key, Inf)
        if dc.d < current_dc
            new_dc_dict[dc_key] = dc.d
        end
    end
    return DCStats(new_def, Set{DC}(DC(key.X, key.Y, d) for (key, d) in new_dc_dict))
end

function merge_tensor_stats_union(op, lstats::DCStats, rstats::DCStats)
    new_def = merge_tensor_def_union(op, get_def(lstats), get_def(rstats))

    # We start by extending both arguments' dcs to the new dimensions
    extended_l_dcs = Set{DC}()
    new_l_indices = setdiff(get_index_set(rstats), get_index_set(lstats))
    for dc in lstats.dcs
        for Z in subset(new_l_indices)
            Z_dimension_space_size = get_dim_space_size(rstats, Z)
            push!(extended_l_dcs, DC(dc.X, ∪(dc.Y, Z), dc.d*Z_dimension_space_size))
        end
    end

    extended_r_dcs = Set{DC}()
    new_r_indices = setdiff(get_index_set(lstats), get_index_set(rstats))
    for dc in rstats.dcs
        for Z in subset(new_r_indices)
            Z_dimension_space_size = get_dim_space_size(lstats, Z)
            push!(extended_r_dcs, DC(dc.X, ∪(dc.Y, Z), dc.d*Z_dimension_space_size))
        end
    end

    # Once the DCs are properly extended, we can add them to get the result
    new_dc_dict = Dict()
    for dc in ∪(rstats.dcs, lstats.dcs)
        dc_key = (X=dc.X, Y=dc.Y)
        current_dc = get(new_dc_dict, dc_key, 0)
        new_dc_dict[dc_key] = current_dc + dc.d
    end

    return DCStats(new_def, new_cardinality)
end

function reduce_tensor_stats(op, reduce_indices::Set{IndexExpr}, stats::DCStats)
    new_def = reduce_tensor_def(op, reduce_indices, get_def(stats))
    # In order to preserve as much information as possible, we need to infer dcs before
    # filtering out DCs.
    inferred_dcs = _infer_dcs(stats.dcs)
    new_dcs_dict = Dict()
    for dc in inferred_dcs
        summed_Xs = ∩(dc.X, reduce_indices)
        remaining_Xs = setdiff(dc.X, reduce_indices)
        summed_X_dim_size = get_dim_space_size(stats, summed_Xs)
        new_Y = setdiff(dc.Y, reduce_indices)
        new_Y_dim_size = get_dim_space_size(stats, new_Y)
        new_dc = min(dc.d * summed_X_dim_size, new_Y_dim_size)
        new_key = (X = remaining_Xs, Y = new_Y)
        current_dc = get(new_dcs_dict, new_key, Inf)
        if new_dc < current_dc
            new_dcs_dict[new_key] = new_dc
        end
    end
    new_dcs = Set{DC}()
    for (key, d) in new_dcs_dict
        push!(new_dcs, DC(key.X, key.Y, d))
    end
    return DCStats(new_def, new_dcs)
end
