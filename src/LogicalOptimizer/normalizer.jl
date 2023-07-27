
function add_global_order(x::LogicalPlanNode, global_index_order)
    return LogicalPlanNode(x.head, [x.args..., global_index_order], nothing)
end

function insert_global_orders(expr, global_index_order)
    global_order_rule = @rule ~x => add_global_order(x, global_index_order) where (x isa LogicalPlanNode && x.head == InputTensor)
    global_order_rule = Metatheory.Postwalk(Metatheory.PassThrough(global_order_rule))
    return global_order_rule(expr)
end

needs_reorder(expr, index_order) = false
function needs_reorder(expr::LogicalPlanNode, index_order)
    return expr.head == InputTensor && is_sorted_wrt_index_order(expr.args[1], index_order)
end

function insert_input_reorders(expr, global_index_order)
    reorder_rule = @rule ~x => Reorder(x, global_index_order) where (needs_reorder(x, global_index_order))
    reorder_rule = Metatheory.Postwalk(Metatheory.PassThrough(reorder_rule))
    return reorder_rule(expr)
end

function remove_uneccessary_reorders(expr, global_index_order)
    if expr.head == Reorder && is_sorted_wrt_index_order(expr.args[2], global_index_order)
        return expr.args[1]
    end
    return expr
end

function merge_aggregates(expr)
    merge_rule = @rule op idx_1 idx_2 x  Aggregate(op, idx_1, Aggregate(op, idx_2, x)) => Aggregate(op, union(idx_1, idx_2), x)
    merge_rule = Metatheory.Postwalk(Metatheory.PassThrough(merge_rule))
    new_expr = merge_rule(expr)
    if new_expr === nothing
        return expr
    else
        return new_expr
    end
end

# This function handles the renaming of indices when expressions are re-used, e.g.
# C[i,j] = A[i,j] * B[i,j]
# D[m, n] = C[m, n] * C[n, m]
# Here, the second indexing of C initially inserts "Rename" operators. This function would
# take the expression tree starting at D and remove the rename operators by translating the
# indices present in lower levels of the expression tree.
function recursive_rename(expr::LogicalPlanNode, index_lookup, depth, context, context_counter, drop_stats, drop_index_order)
    if expr.head == RenameIndices
        expr_index_lookup = Dict()
        renamed_indices::Vector{String} = expr.args[2]
        for i in 1:length(expr.stats.indices)
            new_index = renamed_indices[i]
            if new_index in keys(index_lookup)
                new_index = index_lookup[new_index]
            end
            expr_index_lookup[expr.args[1].stats.indices[i]] = expr.args[2][i]
        end
        context_counter[1] += 1
        context = context_counter[1]
        return recursive_rename(expr.args[1], expr_index_lookup, depth+1, context, context_counter, drop_stats, drop_index_order)
    elseif expr.head == InputTensor
        indices = Vector{String}()
        for index in expr.stats.indices
            if index in keys(index_lookup)
                push!(indices, index_lookup[index])
            elseif context > 0
                push!(indices, index * "_" * string(context))
            else
                push!(indices, index)
            end
        end
        new_args = [indices, expr.args[2]]
        !drop_index_order && push!(new_args, expr.args[3])
        if drop_stats
            return LogicalPlanNode(InputTensor, new_args, nothing)
        else
            return LogicalPlanNode(InputTensor, new_args, expr.stats)
        end
    elseif expr.head == Reorder

        if depth > 0
            return recursive_rename(expr.args[1], index_lookup, depth+1, context, context_counter, drop_stats, drop_index_order)
        end

        new_args = [recursive_rename(expr.args[1], index_lookup, depth+1, context, context_counter, drop_stats, drop_index_order), expr.args[2]]
        if drop_stats
            return LogicalPlanNode(Reorder, new_args, nothing)
        else
            return LogicalPlanNode(Reorder, new_args, expr.stats)
        end
    end

    new_args = []
    for arg in expr.args
        if arg isa LogicalPlanNode
            push!(new_args, recursive_rename(arg, index_lookup, depth + 1, context, context_counter, drop_stats, drop_index_order))
        elseif arg isa Vector{String}
            new_indices = Vector{String}()
            for index in arg
                if index in keys(index_lookup)
                    push!(new_indices, index_lookup[index])
                elseif context > 0
                    push!(new_indices, index * "_" * string(context))
                else
                    push!(new_indices, index)
                end
            end
            push!(new_args, new_indices)
        else
            push!(new_args, arg)
        end
    end

    if drop_stats
        return LogicalPlanNode(expr.head, new_args, nothing)
    else
        return LogicalPlanNode(expr.head, new_args, expr.stats)
    end
end
