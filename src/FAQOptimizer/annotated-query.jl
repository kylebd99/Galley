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

function Base.copy(aq::AnnotatedQuery)
    new_point_expr = plan_copy(aq.point_expr)
    id_to_node = Dict()
    for node in PreOrderDFS(new_point_expr)
        id_to_node[node.node_id] = node
    end
    return AnnotatedQuery(aq.ST,
                          deepcopy(aq.output_name),
                          deepcopy(aq.output_order),
                          deepcopy(aq.output_format),
                          deepcopy(aq.reduce_idxs),
                          new_point_expr,
                          deepcopy(aq.idx_lowest_root),
                          deepcopy(aq.idx_op),
                          id_to_node,
                          deepcopy(aq.parent_idxs),
                          )
end
# Takes in a query and preprocesses it to gather relevant info
# Assumptions:
#      - expr is of the form Query(name, Materialize(formats, index_order, agg_map_expr))
#      - or of the form Query(name, agg_map_expr)
function AnnotatedQuery(q::PlanNode, ST)
    if !(@capture q Query(~name, ~expr))
        throw(ErrorException("Annotated Queries can only be built from queries of the form: Query(name, Materialize(formats, index_order, agg_map_expr)) or Query(name, agg_map_expr)"))
    end
    insert_statistics!(ST, q)
    q = canonicalize(q)
    output_name = q.name
    has_mat_expr = q.expr.kind === Materialize
    expr, output_formats, output_index_order = (nothing, nothing, nothing)
    if has_mat_expr
        mat_expr = q.expr
        output_formats = mat_expr.formats
        output_index_order = mat_expr.idx_order
        expr = mat_expr.expr
    else
        expr = q.expr
    end
    starting_reduce_idxs = []
    idx_starting_root = Dict{PlanNode, Int}()
    idx_op = Dict()
    point_expr = Rewrite(Postwalk(Chain([
        (@rule Aggregate(~f::isvalue, ~idxs..., ~a) => begin
        for idx in idxs
            idx_starting_root[idx] = a.node_id
            idx_op[idx] = f.val
        end
        append!(starting_reduce_idxs, idxs)
        a
        end),
    ])))(expr)
    point_expr = plan_copy(point_expr) # Need to sanitize
    insert_statistics!(ST, point_expr)

    id_to_node = Dict()
    for node in PreOrderDFS(point_expr)
        id_to_node[node.node_id] = node
    end

    reduce_idxs = []
    idx_lowest_root = Dict()
    for idx in starting_reduce_idxs
        agg_op = idx_op[idx]
        idx_dim_size = get_dim_size(point_expr.stats, idx.name)
        lowest_roots = find_lowest_roots(agg_op, idx, id_to_node[idx_starting_root[idx]])
        if length(lowest_roots) == 1
            idx_lowest_root[idx] = only(lowest_roots)
            push!(reduce_idxs, idx)
        else
            new_idxs = [gensym(idx.name) for _ in lowest_roots]
            new_idxs[1] = idx.name
            for i in eachindex(lowest_roots)
                node = id_to_node[lowest_roots[i]]
                if idx.name ∉ get_index_set(node.stats)
                    # If the lowest root doesn't contain the reduction index, we attempt
                    # to remove the reduction via a repeat_operator, i.e.
                    # ∑_i B_j = B_j*|Dom(i)|
                    if isnothing(repeat_operator(agg_op))
                        continue
                    else
                        f = repeat_operator(agg_op)
                        dim_val = Value(idx_dim_size)
                        dim_val.stats = ST(idx_dim_size)
                        dim_val.node_id = maximum(keys(id_to_node)) + 1
                        id_to_node[dim_val.node_id] = dim_val
                        new_node = MapJoin(f, node, dim_val)
                        new_node.stats = merge_tensor_stats(f, node.stats, dim_val.stats)
                        new_node.node_id = node.node_id
                        new_node.node_id = maximum(keys(id_to_node)) + 1
                        id_to_node[new_node.node_id] = new_node
                        point_expr = replace_and_remove_nodes!(point_expr, node.node_id, new_node, [])
                        continue
                    end
                end
                new_idx = new_idxs[i]
                relabel_index(node, idx.name, new_idx)
                idx_op[Index(new_idx)] = agg_op
                idx_starting_root[Index(new_idx)] = idx_starting_root[idx]
                idx_lowest_root[Index(new_idx)] = lowest_roots[i]
                push!(reduce_idxs, Index(new_idx))
            end
        end
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
    if root_node.kind === MapJoin && isdistributive(root_node.op.val, reduce_op)
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
    query_expr = Aggregate(aq.idx_op[reduce_idx], idxs_to_be_reduced..., query_expr)
    query_expr.stats = reduce_tensor_stats(query_expr.op.val, Set([idx.name for idx in idxs_to_be_reduced]), query_expr.arg.stats)
    query = Query(Alias(gensym("A")), query_expr)
    @assert length(∩([idx.name for idx in query_expr.idxs], get_index_set(query_expr.stats))) == 0
    return query, node_to_replace, nodes_to_remove
end

# Returns the cost of reducing out an index
function cost_of_reduce(reduce_idx, aq, cache=Dict())
    query, _, _ = get_reduce_query(reduce_idx, aq)
    cache_key = hash(query.expr)
    if !haskey(cache, cache_key)
        comp_stats = query.expr.arg.stats
        mat_stats = query.expr.stats
        cost = estimate_nnz(comp_stats) * ComputeCost + estimate_nnz(mat_stats) * AllocateCost
        cache[cache_key] = cost
    end
    return cache[cache_key], query.expr.idxs
end

function replace_and_remove_nodes!(expr, node_id_to_replace, new_node, nodes_to_remove)
    if expr.node_id == node_id_to_replace
        return new_node
    end
    for node in PostOrderDFS(expr)
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
    condense_stats!(query.expr.stats)

    reduced_idxs = query.expr.idxs
    alias_expr = Alias(query.name.name)
    alias_expr.node_id = node_to_replace
    alias_expr.stats = deepcopy(query.expr.stats)
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

    insert_statistics!(aq.ST, new_point_expr)
    @assert idx.name ∉ get_index_set(new_point_expr.stats)
    @assert length(unique(aq.reduce_idxs)) == length(aq.reduce_idxs)
    @assert length(unique(new_reduce_idxs)) == length(new_reduce_idxs)
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
    insert_statistics!(aq.ST, expr)
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

# Given a node in the tree, return all indices which can be reduced after computing that subtree.
function get_reducible_idxs(aq, n)
    reduce_idxs = Set()
    for idx in aq.reduce_idxs
        idx_root = aq.idx_lowest_root[idx]
        if intree(idx_root, n)
            push!(reduce_idxs, idx)
        end
    end
    return reduce_idxs
end

# Returns the lowest set of nodes that the reduction can be pushed to
function find_lowest_roots(op, idx, root)
    if @capture root MapJoin(~f, ~args...)
        args_with_idx = [arg for arg in args if idx.name in get_index_set(arg.stats)]
        args_without_idx = [arg for arg in args if idx.name ∉ get_index_set(arg.stats)]
        if isdistributive(f.val, op) && length(args_with_idx) == 1
            return find_lowest_roots(op, idx, only(args_with_idx))
        elseif cansplitpush(f.val, op)
            return [[arg.node_id for arg in args_without_idx]...,
                    reduce(vcat, [find_lowest_roots(op, idx, arg) for arg in args_with_idx])...]
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
