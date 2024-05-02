
mutable struct AnnotatedQuery
    ST
    output_name
    output_order
    output_format
    reduce_idxs
    point_expr
    idx_lowest_root
    idx_op
    id_to_node
    parent_idxs
end

# Takes in a query and preprocesses it to gather relevant info
# Assumptions:
#      - expr is of the form Query(name, Materialize(formats, index_order, agg_map_expr))
function AnnotatedQuery(q::PlanNode, ST)
    if !(@capture q Query(~name, Materialize(~formats..., ~index_order..., ~agg_map_expr)))
        throw(ErrorException("Annotated Queries can only be built from queries of the form: Query(name, Materialize(formats, index_order, agg_map_expr))"))
    end
    insert_statistics!(ST, q)
    q = canonicalize(q)
    insert_statistics!(ST, q)
    output_name = q.name
    mat_expr = q.expr
    output_formats = mat_expr.formats
    output_index_order = mat_expr.idx_order
    expr = mat_expr.expr
    reduce_idxs = []
    idx_starting_root = Dict()
    idx_op = Dict()
    point_expr = Rewrite(Postwalk(Chain([
        (@rule Aggregate(~f::isvalue, ~idxs..., ~a) => begin
        for idx in idxs
            idx_starting_root[idx] = a.node_id
            idx_op[idx] = f.val
        end
        append!(reduce_idxs, idxs)
        a
        end),
    ])))(expr)

    id_to_node = Dict()
    for node in PreOrderDFS(point_expr)
        id_to_node[node.node_id] = node
    end
    idx_lowest_root = Dict()
    for idx in reduce_idxs
        # TODO: When we implement pushing addition over addition, we should rename variables when the aggregate is split.
        # This way, each variable has a unique lowest root.
        idx_lowest_root[idx] = only(find_lowest_roots(idx_op[idx], idx, id_to_node[idx_starting_root[idx]]))
    end
    parent_idxs = Dict(i=>[] for i in reduce_idxs)
    for idx1 in reduce_idxs
        idx1_op = idx_op[idx1]
        idx1_starting_root = id_to_node[idx_starting_root[idx1]]
        idx1_bottom_root = id_to_node[max(idx_lowest_root[idx1]...)]
        for idx2 in reduce_idxs
            idx2_op = idx_op[idx2]
            idx2_starting_root = id_to_node[idx_starting_root[idx2]]
            # if idx1 isn't a parent of idx2, then idx2 can't restrict the summation of idx1
            if isdescendant(idx2_starting_root, idx1_starting_root)
                if idx1_op != idx2_op || !isassociative(idx1_op) || !iscommutative(idx1_op)
                    push!(parent_idxs[idx1], idx2)
                    continue
                end
                if isdescendant(idx2_starting_root, idx1_bottom_root)
                    push!(parent_idxs[idx1], idx2)
                    continue
                end
            end
        end
    end
    return AnnotatedQuery(ST,
                            output_name,
                            output_index_order,
                            output_formats,
                            reduce_idxs,
                            point_expr,
                            idx_lowest_root,
                            idx_op,
                            id_to_node,
                            parent_idxs)
end

function get_reduce_query(reduce_idx, aq)
    reduce_op = aq.idx_op[reduce_idx]
    root_node_id = aq.idx_lowest_root[reduce_idx]
    root_node = aq.id_to_node[root_node_id]
    query_expr = nothing
    idxs_to_be_reduced = Set([reduce_idx])
    nodes_to_remove = Set()
    node_to_replace = -1
    reducible_idxs = get_reducible_idxs(aq)
    if root_node.kind === MapJoin && isdistributive(reduce_op, root_node.op.val)
        # If you're already reducing one index, then it may make sense to reduce others as well.
        # E.g. when you reduce one vertex of a triangle, you should do the other two as well.
        args_with_reduce_idx = [arg for arg in root_node.args if reduce_idx.name in get_index_set(arg.stats)]
        kernel_idxs = union([get_index_set(arg.stats) for arg in args_with_reduce_idx]...)
        relevant_args = [arg for arg in root_node.args if get_index_set(arg.stats) ⊆ kernel_idxs]
        if length(relevant_args) == length(root_node.args)
            node_to_replace = root_node.node_id
        else
            node_to_replace = relevant_args[1].node_id
            for arg in relevant_args[2:end]
                push!(nodes_to_remove, arg.node_id)
            end
        end
        query_expr = MapJoin(root_node.op, relevant_args...)
        query_expr.stats = merge_tensor_stats(root_node.op, [arg.stats for arg in relevant_args]...)

        for idx in reducible_idxs
            if aq.idx_op[idx] != aq.idx_op[reduce_idx]
                continue
            end
            idx_root_id = aq.idx_lowest_root[idx]
            idx_root_node = aq.id_to_node[idx_root_id]
            args_with_idx = [arg for arg in root_node.args if idx.name in get_index_set(arg.stats)]
            if idx_root_id == root_node_id && relevant_args ⊇ args_with_idx
                push!(idxs_to_be_reduced, idx)
            elseif any([intree(idx_root_node, arg) for arg in relevant_args])
                push!(idxs_to_be_reduced, idx)
            end
        end
    else
        query_expr = root_node
        node_to_replace = root_node.node_id
        reducible_idxs = get_reducible_idxs(aq)
        for idx in reducible_idxs
            if aq.idx_op[idx] != aq.idx_op[reduce_idx]
                continue
            end
            idx_root = aq.idx_lowest_root[idx]
            if idx_root == root_node_id
                push!(idxs_to_be_reduced, idx)
            elseif isdescendant(idx_root, root_node)
                push!(idxs_to_be_reduced, idx)
            end
        end
    end
    condense_stats!(query_expr.stats)
    query_expr = Aggregate(aq.idx_op[reduce_idx], idxs_to_be_reduced..., query_expr)
    query_expr.stats = reduce_tensor_stats(query_expr.op, Set(query_expr.idxs), query_expr.arg.stats)
    query = Query(Alias(gensym("A")), query_expr)
    return query, node_to_replace, nodes_to_remove
end

# Returns the cost of reducing out an index
function cost_of_reduce(reduce_idx, aq)
    query, _, _ = get_reduce_query(reduce_idx, aq)
    comp_stats = query.expr.arg.stats
    mat_stats = query.expr.stats
    return estimate_nnz(comp_stats) * ComputeCost + estimate_nnz(mat_stats) * AllocateCost
end

function replace_and_remove_nodes!(expr, node_id_to_replace, new_node, nodes_to_remove)
    if expr.node_id == node_id_to_replace
        return new_node
    end
    for node in PreOrderDFS(expr)
        if node.kind == Plan || node.kind == Query || node.kind == Aggregate
            throw(ErrorException("There should be no $(node.kind) nodes in a pointwise expression."))
        end
        if node.kind == MapJoin && any([arg.node_id == node_id_to_replace || arg.node_id in nodes_to_remove for arg in node.args])
            new_args = [arg for arg in node.args if !(arg.node_id in nodes_to_remove)]
            node.children = vcat([node.op], new_args)
            for i in eachindex(node.args)
                if node.args[i].node_id == node_id_to_replace
                    node.args[i] = new_node
                end
            end
        end
    end
    return expr
end

# Returns a new AQ where `idx` has been reduced out of the expression
# along with the properly formed query which performs that reduction.
function reduce_idx!(idx, aq)
    query, node_to_replace, nodes_to_remove = get_reduce_query(idx, aq)
    reduced_idxs = query.expr.idxs
    alias_expr = Alias(query.name.name)
    alias_expr.node_id = node_to_replace
    alias_expr.stats = deepcopy(query.expr.stats)
    condense_stats!(alias_expr.stats; cheap=false)
    new_point_expr = replace_and_remove_nodes!(aq.point_expr, node_to_replace, alias_expr, nodes_to_remove)
    new_id_to_node = Dict()
    for node in PreOrderDFS(new_point_expr)
        new_id_to_node[node.node_id] = node
    end
    new_reduce_idxs = filter((x) -> !(x in reduced_idxs), aq.reduce_idxs)
    new_idx_lowest_root = Dict()
    new_idx_op = Dict()
    new_parent_idxs = Dict()
    for idx in keys(aq.idx_lowest_root)
        if idx in reduced_idxs
            continue
        end
        root = aq.idx_lowest_root[idx]
        if root == node_to_replace || root ∈ nodes_to_remove
            root = alias_expr.node_id
        end
        new_idx_lowest_root[idx] = root
        new_idx_op[idx] = aq.idx_op[idx]
        new_parent_idxs[idx] = filter((x)->!(x in reduced_idxs), aq.parent_idxs[idx])
    end
    aq.reduce_idxs = new_reduce_idxs
    aq.point_expr = new_point_expr
    aq.idx_lowest_root = new_idx_lowest_root
    aq.idx_op = new_idx_op
    aq.id_to_node = new_id_to_node
    aq.parent_idxs = new_parent_idxs
    return query
end

function get_remaining_query(aq)
    expr = aq.point_expr
    if expr.kind === Alias
        return nothing
    end
    condense_stats!(expr.stats; cheap=false)
    query = Query(aq.output_name, expr)
    insert_statistics!(aq.ST, query)
    return query
end

# Returns the set of indices which are available to be reduced immediately.
function get_reducible_idxs(aq)
    reducible_idxs = [idx for idx in aq.reduce_idxs if length(aq.parent_idxs[idx]) == 0]
    return reducible_idxs
end

# Returns the lowest set of nodes that the reduction can be pushed to
function find_lowest_roots(op, idx, root)
    if @capture root MapJoin(~f, ~args...)
        args_with_idx = [arg for arg in args if idx.name in get_index_set(arg.stats)]
        if isdistributive(op, f.val) && length(args_with_idx) == 1
            return find_lowest_roots(op, idx, only(args_with_idx))
        else
            return [root.node_id]
        end
    elseif root.kind === Alias
        return [root.node_id]
    elseif root.kind === Input
        return [root.node_id]
    else
        throw(ErrorException("There shouldn't be nodes of kind $(root.kind) during root pushdown."))
    end
end