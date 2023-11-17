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

function rename_index(index::IndexExpr, context::Int)
    return IndexExpr(index.id + context, index.name * "_" * string(context))
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
        renamed_indices::Vector{IndexExpr} = expr.args[2]
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
        indices = Vector{IndexExpr}()
        for index in expr.stats.indices
            if index in keys(index_lookup)
                push!(indices, index_lookup[index])
            elseif context > 0
                push!(indices, rename_index(index, context))
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
        elseif arg isa Vector{IndexExpr}
            new_indices = Vector{IndexExpr}()
            for index in arg
                if index in keys(index_lookup)
                    push!(new_indices, index_lookup[index])
                elseif context > 0
                    push!(new_indices, rename_index(index, context))
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
