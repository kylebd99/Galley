using Finch: lift_subqueries, isolate_reformats, propagate_copy_queries
using Finch: propagate_transpose_queries, propagate_map_queries, flatten_plans
using Finch: push_fields, lift_fields


#immediate =  0ID
#deferred  =  1ID
#field     =  2ID
#alias     =  3ID
#table     =  4ID | IS_TREE
#mapjoin   =  5ID | IS_TREE
#aggregate =  6ID | IS_TREE
#reorder   =  7ID | IS_TREE
#relabel   =  8ID | IS_TREE
#reformat  =  9ID | IS_TREE
#subquery  = 10ID | IS_TREE
#query     = 11ID | IS_TREE | IS_STATEFUL
#produces  = 12ID | IS_TREE | IS_STATEFUL
#plan      = 13ID | IS_TREE | IS_STATEFUL

function apply_relabel(idx, bindings)
    return get(bindings, idx, idx)
end

function push_down_relabel(prgm::LogicNode, bindings=Dict())
    if prgm.kind == relabel
        if prgm.arg.kind == alias
            return relabel(prgm.arg, [apply_relabel(idx, bindings) for idx in prgm.idxs]...)
        end
        new_bindings = copy(bindings)
        old_order = FinchLogic.getfields(prgm.arg)
        new_order = prgm.idxs
        for i in eachindex(old_order)
            if haskey(new_bindings, new_order[i])
                new_bindings[old_order[i]] = new_bindings[new_order[i]]
                delete!(new_bindings, new_order[i])
            else
                new_bindings[old_order[i]] = new_order[i]
            end
        end
        return push_down_relabel(prgm.arg, new_bindings)
    elseif prgm.kind == field
        return apply_relabel(prgm, bindings)
    elseif istree(prgm)
        prgm.children = LogicNode[push_down_relabel(c, bindings) for c in prgm.children]
        return prgm
    else
        return prgm
    end
end

function pull_up_reorders(prgm::LogicNode)
    
end

function unwrap_subqueries(prgm::LogicNode)
    Rewrite(Postwalk(Chain([@rule subquery(~lhs, ~rhs)=>rhs])))(prgm)
end

function normalize_hl(prgm::LogicNode)
    #deduplicate and lift inline subqueries to regular queries
    prgm = unwrap_subqueries(prgm)
    prgm = flatten_plans(prgm)
    prgm = propagate_fields(prgm)
    prgm = push_down_relabel(prgm)
#    prgm = push_fields(prgm)
#    prgm = lift_fields(prgm)
#    prgm = push_fields(prgm)

#    prgm = propagate_copy_queries(prgm)
#    prgm = propagate_transpose_queries(prgm)
#    prgm = propagate_map_queries(prgm)
    prgm = flatten_plans(prgm)
    return prgm
end

# Galley takes in a Finch Logic Program with the following form:
# pgrm := plan
# plan := query..., produces
# produces := symbols...
# query := reformat
# reformat := format..., reorder
# reorder := field..., expr
# expr := aggregate | mapjoin | alias | table
# table := symbol, field...
# aggregate := op, init, expr, field...
function finch_hl_to_galley(prgm::LogicNode)
    println(prgm)
    if prgm.kind == plan
        query_nodes = [q for q in prgm.bodies if q.kind == query]
        return PlanNode[finch_hl_to_galley(q) for q in query_nodes]
    elseif prgm.kind == produces
        return # TODO: Change Galley to accept a produces list
    elseif prgm.kind == query
        rhs = finch_hl_to_galley(prgm.rhs)
        if rhs.kind != Materialize 
            rhs = Mat([IndexExpr(i.val) for i in FinchLogic.getfields(prgm.rhs)]..., rhs)
        end
        lhs = finch_hl_to_galley(prgm.lhs)
        return Query(lhs, rhs)
    elseif prgm.kind == reformat
        formats = get_tensor_formats(prgm.tns.val)
        @assert prgm.arg.kind == reorder
        reorder_arg = prgm.arg
        idxs = [IndexExpr(i.val) for i in reorder_arg.idxs]
        return Mat(formats..., idxs..., finch_hl_to_galley(reorder_arg.arg))
    elseif prgm.kind == reorder
        println("HERE")
        if prgm.arg.kind == relabel || prgm.arg.kind == table
            if prgm.idxs == prgm.arg.idxs
                return finch_hl_to_galley(prgm.arg)
            end
            println(prgm.idxs)
            println(prgm.arg.idxs)
            throw(error("Broadcasting not yet implemented in Galley!"))
        end
        idxs = [IndexExpr(i.val) for i in prgm.idxs]
        return Mat(idxs..., finch_hl_to_galley(prgm.arg))
    elseif prgm.kind == aggregate
        return Aggregate(finch_hl_to_galley(prgm.op),
                        [finch_hl_to_galley(i) for i in prgm.idxs]...,
                         finch_hl_to_galley(prgm.arg))
    elseif prgm.kind == mapjoin
        return MapJoin(finch_hl_to_galley(prgm.op),
                        [finch_hl_to_galley(arg) for arg in prgm.args]...)
    elseif prgm.kind == table
        return Input(finch_hl_to_galley(prgm.tns),
                    [finch_hl_to_galley(i) for i in prgm.idxs]...)
    elseif prgm.kind == relabel
        @assert prgm.arg.kind == alias
        return Alias(prgm.arg.val, [finch_hl_to_galley(i) for i in prgm.idxs]...)
    elseif prgm.kind == alias
        return Alias(prgm.val)
    elseif prgm.kind == field
        return IndexExpr(prgm.val)
    elseif prgm.kind == immediate
        return Value(prgm.val)
    else
        throw(error("Finch Logic statements of kind $(prgm.kind) is not supported by Galley."))
    end
end
