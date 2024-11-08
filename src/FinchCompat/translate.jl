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

function push_relabels(prgm)
    prgm = Rewrite(Fixpoint(Prewalk(Chain([
        (@rule relabel(mapjoin(~op, ~args...), ~idxs...) => begin
            idxs_2 = getfields(mapjoin(op, args...))
            mapjoin(op, map(arg -> relabel(reorder(arg, idxs_2...), idxs...), args)...)
            end),
            (@rule relabel(aggregate(~op, ~init, ~arg, ~idxs1...), ~idxs2...) => begin
                arg_idxs = Finch.getfields(arg)
                arg_relabel_idxs = filter((i) -> i ∉ idxs1, arg_idxs)
                relabel_dict = merge(Dict(i=>i for i in idxs1), Dict(arg_relabel_idxs[i]=>idxs2[i] for i in eachindex(idxs2)))
                aggregate(op, init, relabel(arg, [relabel_dict[i] for i in arg_idxs]...), idxs1...)
            end),
            (@rule relabel(relabel(~arg, ~idxs...), ~idxs_2...) =>
                relabel(~arg, ~idxs_2...)),
            (@rule relabel(reorder(~arg, ~idxs_1...), ~idxs_2...) => begin
                idxs_3 = getfields(arg)
                reidx = Dict(map(Pair, idxs_1, idxs_2)...)
                idxs_4 = map(idx -> get(reidx, idx, idx), idxs_3)
                reorder(relabel(arg, idxs_4...), idxs_2...)
            end),
            (@rule relabel(table(~arg, ~idxs_1...), ~idxs_2...) => begin
                table(arg, idxs_2...)
            end),
            (@rule relabel(~arg::isimmediate) => arg),
    ]))))(prgm)

    
end

function push_reorders(prgm::LogicNode)
    prgm = Rewrite(Fixpoint(Prewalk(Chain([
        (@rule reorder(mapjoin(~op, ~args...), ~idxs...) =>
            mapjoin(op, map(arg -> reorder(arg, idxs...), args)...)),
        (@rule reorder(reorder(~arg, ~idxs...), ~idxs_2...) =>
            reorder(arg, idxs_2...) where Set(idxs) == Set(idxs_2)),
    ]))))(prgm)    
    #= 
    prgm = Rewrite(Prewalk(Fixpoint(Chain([
        (@rule reorder(table(~tns, ~idxs1...), ~idxs2...) => table(tns, idxs1...) where length(idxs2) >= length(idxs1)),
        (@rule reorder(table(~tns, ~idxs1...), ~idxs2...) => aggregate(initwrite(nothing), nothing, table(tns, idxs1...), setdiff(idxs1, idxs2)...) where length(idxs2) < length(idxs1)),
    ]))))(prgm) =#
    prgm
end

function aggs_to_mapjoins(prgm)
    Rewrite(Fixpoint(Prewalk(Chain([
        (@rule aggregate(~op, ~init, ~arg) => arg where isnothing(init)),
        (@rule aggregate(~op, ~init, ~arg) => mapjoin(op, init, arg)),
    ]))))(prgm)
end

function compatible_order(order1, order2)
    resulting_order = []
    cur_lpos = 1
    cur_rpos = 1
    while cur_lpos <= length(order1) && cur_rpos <= length(order2)
        if order1[cur_lpos].val == order2[cur_rpos].val
            push!(resulting_order, order1[cur_lpos])
            cur_lpos += 1
            cur_rpos += 1
        elseif order1[cur_lpos] ∉ order2
            push!(resulting_order, order1[cur_lpos])
            cur_lpos += 1
        else
            push!(resulting_order, order2[cur_rpos])
            cur_rpos += 1
        end
    end
    while cur_lpos <= length(order1) 
        push!(resulting_order, order1[cur_lpos])
        cur_lpos += 1
    end
    while cur_rpos <= length(order2) 
        push!(resulting_order, order2[cur_rpos])
        cur_rpos += 1
    end
    return resulting_order
end

function pull_up_reorders(prgm::LogicNode)
    prgm = Rewrite(Fixpoint(Postwalk(Chain([(@rule reorder(~arg, ~idxs2...) => 
                        reorder(aggregate(nothing, nothing, arg, setdiff(getfields(arg), idxs2)...), idxs2...) where length(idxs2) < length(getfields(arg)))]))))(prgm)

    prgm = Rewrite(Fixpoint(Postwalk(Chain([
                    (@rule reorder(reorder(~arg, ~idxs1...), ~idxs2...)=>
                        begin
                            idxs3 = compatible_order(idxs1, idxs2)
                            reorder(arg, idxs3...)
                        end),
                    (@rule aggregate(~op, ~init, reorder(~arg, ~ro_idxs...), ~agg_idxs...)=>
                        begin
                            bc_idxs = setdiff(ro_idxs, getfields(arg))
                            eliminated_idxs = intersect(bc_idxs, agg_idxs)
                            reorder(aggregate(op, init, arg, setdiff(agg_idxs, bc_idxs)...), setdiff(ro_idxs, agg_idxs)...)
                        end),
                    (@rule mapjoin(~op, ~preargs..., reorder(~arg, ~ro_idxs...), ~postargs...)=>
                        reorder(mapjoin(op, preargs..., arg, postargs...), ro_idxs...))]))))(prgm)

    prgm                         
end

function remove_extraneous_reorders(prgm::LogicNode)
    #= 
    useless_idxs = LogicNode[]
    for q in PreOrderDFS(prgm)
        if q.kind == reorder
            existing_fields = Finch.getfields(q.arg)
            if length(existing_fields) < length(q.idxs)
                append!(useless_idxs, q.idxs[length(existing_fields)+1:end])
            end
        end
    end
    println("Useless Idxs: \n",useless_idxs)
    Rewrite(Fixpoint(Postwalk(Chain([(@rule aggregate(~op, ~init, ~arg, ~agg_idxs...)=> aggregate(op, init, arg, setdiff(agg_idxs, useless_idxs)...)),
                            (@rule aggregate(~op, ~init, ~arg)=> arg where isnothing(init.val)),
                            (@rule relabel(~arg, ~idxs...) => relabel(arg, setdiff(idxs, useless_idxs))),
                            (@rule relabel(~arg) => arg),
                            (@rule reorder(~arg, ~idxs...) => reorder(arg, setdiff(idxs, useless_idxs))),
                            (@rule reorder(~arg) => arg)]))))(prgm) 
    Rewrite(Fixpoint(Postwalk(Chain([(@rule reorder(~arg, ~idxs...) => reorder(arg, intersect(idxs, getfields(arg))...)),
                                     (@rule reorder(~arg, ~idxs...) => arg where idxs==getfields(arg))]))))(prgm)=#
end

function unwrap_subqueries(prgm::LogicNode)
    Rewrite(Postwalk(Chain([@rule subquery(~lhs, ~rhs)=>rhs])))(prgm)
end

function normalize_hl(prgm::LogicNode)
    #deduplicate and lift inline subqueries to regular queries
    prgm = unwrap_subqueries(prgm)
    prgm = flatten_plans(prgm)
    prgm = push_relabels(prgm)
    prgm = pull_up_reorders(prgm)
    prgm = aggs_to_mapjoins(prgm)
    prgm = flatten_plans(prgm)
    return prgm
end

function normalize_hl2(prgm::LogicNode)
    #deduplicate and lift inline subqueries to regular queries
    prgm = Finch.lift_subqueries(prgm)


    #these steps lift reformat, aggregate, and table nodes into separate
    #queries, using subqueries as temporaries.
    prgm = Finch.isolate_reformats(prgm)
    
    prgm = Finch.isolate_aggregates(prgm)
    
    prgm = Finch.isolate_tables(prgm)

    prgm = Finch.lift_subqueries(prgm)


    #These steps fuse copy, permutation, and mapjoin statements
    #into later expressions.
    #Only reformat statements preserve intermediate breaks in computation
    prgm = Finch.propagate_copy_queries(prgm)
    prgm = Finch.propagate_transpose_queries(prgm)
    prgm = Finch.propagate_map_queries(prgm)

    #These steps assign a global loop order to each statement.
    prgm = Finch.propagate_fields(prgm)
    prgm = Finch.push_fields(prgm)
    prgm = Finch.lift_fields(prgm)
    prgm = Finch.push_fields(prgm)

    prgm = Finch.flatten_plans(prgm)
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
        idxs = [IndexExpr(i.val) for i in prgm.idxs]
        return Mat(idxs..., finch_hl_to_galley(prgm.arg))
    elseif prgm.kind == aggregate
        return Aggregate(finch_hl_to_galley(prgm.op),
                        finch_hl_to_galley(prgm.init),
                        [finch_hl_to_galley(i) for i in prgm.idxs]...,
                         finch_hl_to_galley(prgm.arg))
    elseif prgm.kind == mapjoin
        return MapJoin(finch_hl_to_galley(prgm.op),
                        [finch_hl_to_galley(arg) for arg in prgm.args]...)
    elseif prgm.kind == table
        if prgm.tns.val isa Tensor
            return Input(prgm.tns.val, [finch_hl_to_galley(i) for i in prgm.idxs]...)
        else
            return Input(Tensor(prgm.tns.val), [finch_hl_to_galley(i) for i in prgm.idxs]...)
        end
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
