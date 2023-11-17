# This file defines how stats are produced for a logical expression based on its children.


# For join-like operations (e.g. multiplication when 0 is the default value), this is how
# we merge tensor stats.
function merge_tensor_stats_join(op, lstats::TensorStats, rstats::TensorStats)
    new_default_value = op(lstats.default_value, rstats.default_value)
    new_index_set = union(lstats.index_set, rstats.index_set)

    new_dim_size = Dict()
    for index in new_index_set
        if index in lstats.index_set && index in rstats.index_set
            # Here, we assume that indices can only be shared between tensors with the same
            # dimensions.
            @assert lstats.dim_size[index] == rstats.dim_size[index]
            new_dim_size[index] = lstats.dim_size[index]
        elseif index in rstats.index_set
            new_dim_size[index] = rstats.dim_size[index]
        else
            new_dim_size[index] = lstats.dim_size[index]
        end
    end
    new_dim_space_size = prod([new_dim_size[x] for x in new_index_set])
    l_dim_space_size = prod([lstats.dim_size[x] for x in lstats.index_set])
    r_dim_space_size = prod([rstats.dim_size[x] for x in rstats.index_set])
    l_prob_non_default = (lstats.cardinality/l_dim_space_size)
    r_prob_non_default = (rstats.cardinality/r_dim_space_size)
    new_cardinality = l_prob_non_default * r_prob_non_default * new_dim_space_size
    return TensorStats(new_index_set, new_dim_size, new_cardinality, new_default_value, nothing)
end

function merge_tensor_stats_union(op, lstats::TensorStats, rstats::TensorStats)
    new_default_value = op(lstats.default_value, rstats.default_value)
    new_index_set = union(lstats.index_set, rstats.index_set)

    new_dim_size = Dict()
    for index in new_index_set
        if index in lstats.index_set && index in rstats.index_set
            # Here, we assume that indices can only be shared between tensors with the same
            # dimensions.
            @assert lstats.dim_size[index] == rstats.dim_size[index]
            new_dim_size[index] = lstats.dim_size[index]
        elseif index in rstats.index_set
            new_dim_size[index] = rstats.dim_size[index]
        else
            new_dim_size[index] = lstats.dim_size[index]
        end
    end

    new_dim_space_size = prod([new_dim_size[x] for x in new_index_set])
    l_dim_space_size = prod([lstats.dim_size[x] for x in lstats.index_set])
    r_dim_space_size = prod([rstats.dim_size[x] for x in rstats.index_set])
    l_prob_default = (1 - lstats.cardinality/l_dim_space_size)
    r_prob_default = (1 - rstats.cardinality/r_dim_space_size)
    new_cardinality = (1 - l_prob_default * r_prob_default) * new_dim_space_size
    return TensorStats(new_index_set, new_dim_size, new_cardinality, new_default_value, nothing)
end

function merge_tensor_stats(op, lstats::TensorStats, rstats::TensorStats)
    if length(lstats.index_set) == 0
        return TensorStats(rstats.index_set, rstats.dim_size, rstats.cardinality, op(lstats.default_value, rstats.default_value), nothing)
    elseif length(rstats.index_set) == 0
        return TensorStats(lstats.index_set, lstats.dim_size, lstats.cardinality, op(lstats.default_value, rstats.default_value), nothing)
    end

    if !haskey(annihilator_dict, :($op))
        return merge_tensor_stats_union(op, lstats, rstats)
    end

    annihilator_value = annihilator_dict[:($op)]
    if annihilator_value == lstats.default_value && annihilator_value == rstats.default_value
        return merge_tensor_stats_join(op, lstats, rstats)
    else
        return merge_tensor_stats_union(op, lstats, rstats)
    end
end

function reduce_tensor_stats(op, reduce_indices::Set{IndexExpr}, stats::TensorStats)
    new_default_value = nothing
    if haskey(identity_dict, :($op)) && identity_dict[:($op)] == stats.default_value
        new_default_value = stats.default_value
    elseif op == +
        new_default_value = stats.default_value * prod([stats.dim_size[x] for x in reduce_indices])
    elseif op == *
        new_default_value = stats.default_value ^ prod([stats.dim_size[x] for x in reduce_indices])
    else
        # This is going to be VERY SLOW. Should raise a warning about reductions over non-identity default values.
        # Depending on the semantics of reductions, we might be able to do this faster.
        println("Warning: A reduction can take place over a tensor whose default value is not the reduction operator's identity. \\
                         This can result in a large slowdown as the new default is calculated.")
        new_default_value = op([stats.default_value for _ in prod([stats.dim_size[x] for x in reduce_indices])]...)
    end

    new_index_set = setdiff(stats.index_set, reduce_indices)
    new_dim_size = Dict()
    for index in new_index_set
        new_dim_size[index] = stats.dim_size[index]
    end


    new_dim_space_size = 1
    if length(new_index_set) > 0
        new_dim_space_size = prod([new_dim_size[x] for x in new_index_set])
    end
    old_dim_space_size = 1
    if length(stats.index_set) > 0
        old_dim_space_size = prod([stats.dim_size[x] for x in stats.index_set])
    end
    prob_default_value = 1 - stats.cardinality/old_dim_space_size
    prob_non_default_subspace = 1 - prob_default_value ^ (old_dim_space_size/new_dim_space_size)
    new_cardinality = new_dim_space_size * prob_non_default_subspace
    return TensorStats(new_index_set, new_dim_size, new_cardinality, new_default_value, nothing)
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
    elseif n.head == InputTensor
        n.stats = TensorStats(n.args[1], n.args[2])
    elseif n.head == Scalar
        TensorStats(n.args[1])
    end
    return n.stats
end
