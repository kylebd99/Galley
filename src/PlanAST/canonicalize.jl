
function merge_mapjoins(plan::PlanNode)
    Rewrite(Postwalk(Chain([
        (@rule MapJoin(~f::isvalue, ~a..., MapJoin(~f, ~b...), ~c...) => MapJoin(~f, ~a..., ~b..., ~c...) where isassociative(f.val)),
        (@rule MapJoin(~f::isvalue, ~a..., MapJoin(~f, ~b...)) => MapJoin(~f, ~a..., ~b...) where isassociative(f.val)),
        (@rule MapJoin(~f::isvalue, MapJoin(~f, ~a...), ~b...) => MapJoin(~f, ~a..., ~b...) where isassociative(f.val)),
    ])))(plan)
end

# In cannonical form, two aggregates in the plan shouldn't reduce out the same variable.
# E.g. MapJoin(*, Aggregate(+, Input(tns, i, j)))
function unique_indices(scope_dict, n::PlanNode)
    if n.kind === Plan
        return Plan([unique_indices(scope_dict, query) for query in n.queries]..., n.outputs)
    elseif n.kind === Query
        return Query(n.name, unique_indices(scope_dict, n.expr))
    elseif n.kind === Materialize
        return Materialize(n.formats, n.idx_order, unique_indices(scope_dict, n.expr))
    elseif n.kind === MapJoin
        return MapJoin(n.op, [unique_indices(scope_dict, arg) for arg in n.args]...)
    elseif n.kind === Input
        return relabel_input(n, [unique_indices(scope_dict, idx).name for idx in n.idxs]...)
    elseif n.kind === Aggregate
        new_scope_dict = deepcopy(scope_dict)
        new_idxs = []
        for idx in n.idxs
            old_idx = idx.val
            new_idx = haskey(new_scope_dict, old_idx) ? gensym(idx.val) : idx.val
            push!(new_idxs, new_idx)
            new_scope_dict[old_idx] = new_idx
        end
        return Aggregate(n.op, new_idxs..., unique_indices(new_scope_dict, n.arg))
    elseif n.kind === Index
        return Index(get(scope_dict, n.name, n.name))
    else
        return n
    end
end

function insert_statistics!(ST, plan::PlanNode; bindings = Dict(), replace=false)
    for expr in PostOrderDFS(plan)
        if @capture expr Query(~a, ~expr)
            bindings[a] = expr.stats
        elseif @capture expr Query(~a, ~expr, ~loop_order...)
            bindings[a] = expr.stats
        elseif @capture expr MapJoin(~f, ~args...)
            expr.stats = merge_tensor_stats(f.val, ST[arg.stats for arg in args]...)
        elseif @capture expr Aggregate(~f, ~idxs..., ~arg)
            expr.stats = reduce_tensor_stats(f.val, Set{IndexExpr}([idx.name for idx in idxs]), arg.stats)
        elseif @capture expr Materialize(~formats..., ~idxs..., ~arg)
            expr.stats = arg.stats
            def = get_def(expr.stats)
            def.level_formats = [f.val for f in expr.formats]
            def.index_order = [idx.name for idx in expr.idx_order]
        elseif expr.kind === Alias
            if isnothing(expr.stats)
                expr.stats = get(bindings, expr, nothing)
            end
        elseif @capture expr Input(~tns, ~idxs...)
            if !isnothing(expr.stats) && !replace
                continue
            end
            expr.stats = ST(tns.val, IndexExpr[idx.val for idx in idxs])
        end
    end
end

# This function labels every node with an id. These ids respect a topological ordering where
# children have id's that are larger than parents.
function insert_node_ids!(plan::PlanNode)
    cur_id = 0
    for expr in PreOrderDFS(plan)
        expr.node_id = cur_id
        cur_id += 1
    end
end


function canonicalize(plan::PlanNode)
    plan = merge_mapjoins(plan)
    plan = unique_indices(Dict(), plan)
    insert_node_ids!(plan)
    return plan
end
