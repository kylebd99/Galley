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
    parent_idxs # Index orders that must be respected
    original_idx # When an index is split into many, we track their relationship.
    connected_components
end

function copy_aq(aq::AnnotatedQuery)
    new_point_expr = plan_copy(aq.point_expr)
    id_to_node = Dict()
    for node in PreOrderDFS(new_point_expr)
        id_to_node[node.node_id] = node
    end
    return AnnotatedQuery(aq.ST,
                          aq.output_name,
                          copy(aq.output_order),
                          copy(aq.output_format),
                          copy(aq.reduce_idxs),
                          new_point_expr,
                          copy(aq.idx_lowest_root),
                          copy(aq.idx_op),
                          id_to_node,
                          copy(aq.parent_idxs),
                          copy(aq.original_idx),
                          copy(aq.connected_components),
                          )
end

function get_idx_connected_components(parent_idxs)
    component_ids = Dict(x=>i for (i,x) in enumerate(keys(parent_idxs)))
    finished = false
    while !finished
        finished = true
        for (idx1, parents) in parent_idxs
            for idx2 in parents
                if component_ids[idx2] != component_ids[idx1]
                    finished = false
                end
                component_ids[idx2] = min(component_ids[idx2], component_ids[idx1])
                component_ids[idx1] = min(component_ids[idx2], component_ids[idx1])
            end
        end
    end
    components = []
    for id in unique(values(component_ids))
        idx_in_component = []
        for idx in keys(parent_idxs)
            if component_ids[idx] == id
                push!(idx_in_component, idx)
            end
        end
        push!(components, idx_in_component)
    end
    return components
end

# Takes in a query and preprocesses it to gather relevant info
# Assumptions:
#      - expr is of the form Query(name, Materialize(formats, index_order, agg_map_expr))
#      - or of the form Query(name, agg_map_expr)
function AnnotatedQuery(q::PlanNode, ST; do_pushdown=true)
    if !(@capture q Query(~name, ~expr))
        throw(ErrorException("Annotated Queries can only be built from queries of the form: Query(name, Materialize(formats, index_order, agg_map_expr)) or Query(name, agg_map_expr)"))
    end
    insert_node_ids!(q)
    insert_statistics!(ST, q)
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
    # This dictionary captures the original topological ordering of the aggregates.
    idx_top_order = Dict{PlanNode, Int}()
    top_counter = 1
    idx_op = Dict()
    point_expr = Rewrite(Postwalk(Chain([
        (@rule Aggregate(~f::isvalue, ~idxs..., ~a) => begin
        for idx in idxs
            idx_starting_root[idx] = a.node_id
            idx_top_order[idx] = top_counter
            top_counter += 1
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
    original_idx = Dict(idx => idx for idx in get_index_set(q.expr.stats))
    idx_lowest_root = Dict()
    for idx in starting_reduce_idxs
        agg_op = idx_op[idx]
        idx_dim_size = get_dim_size(point_expr.stats, idx.name)
        lowest_roots = find_lowest_roots(agg_op, idx, id_to_node[idx_starting_root[idx]], do_pushdown)
        original_idx[idx.name] = idx.name
        if length(lowest_roots) == 1
            idx_lowest_root[idx] = only(lowest_roots)
            push!(reduce_idxs, idx)
        else
            new_idxs = [gensym(idx.name) for _ in lowest_roots]
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
                idx_op[Index(new_idx)] = agg_op
                idx_lowest_root[Index(new_idx)] = lowest_roots[i]
                idx_starting_root[Index(new_idx)] = idx_starting_root[idx]
                #TODO: This forces us to do pushdown, limiting us from valuable options.
                idx_starting_root[Index(new_idx)] = lowest_roots[i]
                original_idx[new_idx] = idx.name
                push!(reduce_idxs, Index(new_idx))
            end
        end
    end

    parent_idxs = Dict(i=>[] for i in reduce_idxs)
    connected_idxs = Dict(i=>[] for i in reduce_idxs)
    for idx1 in reduce_idxs
        idx1_op = idx_op[idx1]
        idx1_bottom_root = id_to_node[idx_lowest_root[idx1]]
        for idx2 in reduce_idxs
            idx2_op = idx_op[idx2]
            idx2_top_root = id_to_node[idx_starting_root[idx2]]
            idx2_bottom_root = id_to_node[idx_lowest_root[idx2]]
            if intree(idx2_bottom_root, idx1_bottom_root)
                push!(connected_idxs[idx1], idx2)
            end
            mergeable_agg_op = (idx1_op == idx2_op && isassociative(idx1_op) && iscommutative(idx1_op))
            # If idx1 isn't a parent of idx2, then idx2 can't restrict the summation of idx1
            if isdescendant(idx2_top_root, idx1_bottom_root)
                push!(parent_idxs[idx1], idx2)
            # If they can both be pushed past each other, then we check whether they
            # commute. If not, we check which one was lower in the topological order.
            elseif (!(mergeable_agg_op) &&
                        idx_top_order[idx2] < idx_top_order[idx1])
                push!(parent_idxs[idx1], idx2)
            end
        end
    end

    # If a set of aggregates are unrelated in the expression tree, then they don't need to
    # be co-optimized.
    connected_components = get_idx_connected_components(connected_idxs)
    return AnnotatedQuery(ST,
                            output_name,
                            output_index_order,
                            output_formats,
                            reduce_idxs,
                            point_expr,
                            idx_lowest_root,
                            idx_op,
                            id_to_node,
                            parent_idxs,
                            original_idx,
                            connected_components)
end

function get_reduce_query(reduce_idx, aq)
    original_idx = aq.original_idx[reduce_idx.name]
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
        args_with_reduce_idx = [arg for arg in root_node.args if original_idx in get_index_set(arg.stats)]
        kernel_idxs = union([get_index_set(arg.stats) for arg in args_with_reduce_idx]...)
        relevant_args = [arg for arg in root_node.args if get_index_set(arg.stats) ⊆ kernel_idxs]
        if length(relevant_args) == length(root_node.args)
            node_to_replace = root_node.node_id
            for node in PreOrderDFS(root_node)
                if node.node_id != node_to_replace
                    push!(nodes_to_remove, node.node_id)
                end
            end
        else
            node_to_replace = relevant_args[1].node_id
            for arg in relevant_args[2:end]
                for node in PreOrderDFS(arg)
                    push!(nodes_to_remove, node.node_id)
                end
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
            args_with_idx = [arg for arg in root_node.args if aq.original_idx[idx.name] in get_index_set(arg.stats)]
            if (idx_root_id == root_node_id) && (relevant_args ⊇ args_with_idx)
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
            idx_root_node = aq.id_to_node[idx_root]
            if idx_root == root_node_id
                push!(idxs_to_be_reduced, idx)
            elseif isdescendant(idx_root_node, root_node)
                push!(idxs_to_be_reduced, idx)
            end
        end
    end
    query_expr = plan_copy(query_expr)
    final_idxs_to_be_reduced = Set()
    for idx in idxs_to_be_reduced
        if aq.original_idx[idx.name] != idx.name
            push!(final_idxs_to_be_reduced, Index(aq.original_idx[idx.name]))
        else
            push!(final_idxs_to_be_reduced, idx)
        end
    end

    reduced_idxs = idxs_to_be_reduced
    query_expr = Aggregate(aq.idx_op[reduce_idx], final_idxs_to_be_reduced..., query_expr)
    query_expr.stats = reduce_tensor_stats(query_expr.op.val, Set([idx.name for idx in final_idxs_to_be_reduced]), query_expr.arg.stats)
    query = Query(Alias(gensym("A")), query_expr)
    @assert length(∩([idx.name for idx in query_expr.idxs], get_index_set(query_expr.stats))) == 0
    return query, node_to_replace, nodes_to_remove, reduced_idxs
end

function get_forced_transpose_cost(n)
    inputs = get_inputs(n)
    aliases = get_aliases(n)
    if length(inputs) == 0
        return 0
    end
    vertices = union([get_index_set(input.stats) for input in inputs]..., [get_index_set(alias.stats) for alias in aliases]...)
    vertex_graph = Dict(v => [] for v in vertices)
    for input in inputs
        idx_order = get_index_order(input.stats)
        for i in eachindex(idx_order)
            i+1 > length(idx_order) && continue
            for j in (i+1):length(idx_order)
                push!(vertex_graph[idx_order[i]], idx_order[j])
            end
        end
    end
    # Aliases don't have a defined order, so we include both directions for each edge.
    for alias in aliases
        for i in get_index_set(alias.stats)
            for j in get_index_set(alias.stats)
                if i != j
                    push!(vertex_graph[i], j)
                end
            end
        end
    end

    sinks = [v for v in vertices if length(vertex_graph[v]) == 0]
    if length(sinks) > 1
        return maximum([estimate_nnz(input.stats) for input in inputs if length(input.idxs) > 1]; init=0) * AllocateCost
    else
        return 0
    end
end

function is_dense(mat_stats)
    sparsity = estimate_nnz(mat_stats) / get_dim_space_size(mat_stats, get_index_set(mat_stats))
    return sparsity > .5
end

# Returns the cost of reducing out an index
function cost_of_reduce(reduce_idx, aq, cache=Dict(), alias_hash=Dict())
    query, _, _,_ = get_reduce_query(reduce_idx, aq)
    cache_key = cannonical_hash(query.expr, alias_hash)
    if !haskey(cache, cache_key)
        comp_stats = query.expr.arg.stats
        mat_stats = query.expr.stats
        mat_factor = is_dense(mat_stats)  ? DenseAllocateCost : SparseAllocateCost
        comp_factor = length(get_index_set(comp_stats)) * ComputeCost
        cost = estimate_nnz(comp_stats) * comp_factor + estimate_nnz(mat_stats) * mat_factor
        forced_transpose_cost = get_forced_transpose_cost(query.expr)
        if count_index_occurences([query.expr.arg]) > 10
            cost += count_index_occurences([query.expr]) * cost # We really would rather not split.
        end
        if cost == Inf
            println("INFINITE QUERY FOR: $reduce_idx")
            println(query)
            println("COMP STATS: $(estimate_nnz(comp_stats))")
            println(comp_stats)
            println("MAT STATS: $(estimate_nnz(mat_stats))")
            println(mat_stats)
        end
        cache[cache_key] = cost + forced_transpose_cost
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
function reduce_idx!(reduce_idx, aq)
    query, node_to_replace, nodes_to_remove, reduced_idxs = get_reduce_query(reduce_idx, aq)
#    condense_stats!(query.expr.stats)

    alias_expr = Alias(query.name.name)
    alias_expr.node_id = node_to_replace
    alias_expr.stats = copy_stats(query.expr.stats)
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
    reduced_idx_exprs = Set{IndexExpr}([idx.name for idx in reduced_idxs])
    insert_statistics!(aq.ST, new_point_expr)
    @assert all([idx ∉ get_index_set(new_point_expr.stats) for idx in reduced_idx_exprs])
    @assert length(unique(aq.reduce_idxs)) == length(aq.reduce_idxs)
    @assert length(unique(new_reduce_idxs)) == length(new_reduce_idxs)
    @assert all([haskey(new_id_to_node, new_idx_lowest_root[idx]) for idx in new_reduce_idxs])
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
    reduce_idxs = Set{IndexExpr}()
    for idx in aq.reduce_idxs
        idx_root = aq.idx_lowest_root[idx]
        if intree(idx_root, n)
            push!(reduce_idxs, idx)
        end
    end
    return reduce_idxs
end

# Returns the lowest set of nodes that the reduction can be pushed to
function find_lowest_roots(op, idx, root, do_pushdown)
    if @capture root MapJoin(~f, ~args...)
        args_with_idx = [arg for arg in args if idx.name in get_index_set(arg.stats)]
        args_without_idx = [arg for arg in args if idx.name ∉ get_index_set(arg.stats)]
        if isdistributive(f.val, op) && length(args_with_idx) == 1
            return find_lowest_roots(op, idx, only(args_with_idx), do_pushdown)
        elseif do_pushdown && cansplitpush(f.val, op)
            return [[arg.node_id for arg in args_without_idx]...,
                    reduce(vcat, [find_lowest_roots(op, idx, arg, do_pushdown) for arg in args_with_idx])...]
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
