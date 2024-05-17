MAX_INDEX_OCCURENCES = 8

function count_index_occurences(nodes)
    occurences = 0
    for n in nodes
        for c in PostOrderDFS(n)
            if c.kind == Input
                occurences += length(c.idxs)
            elseif c.kind == Alias
                occurences += length(get_index_set(c.stats))
            end
        end
    end
    return occurences
end

function get_connected_subsets(nodes)
    ss = []
    for k in 2:4
        for s in subsets(nodes, k)
            if count_index_occurences(s) > MAX_INDEX_OCCURENCES
                continue
            end
            is_cross_prod = false
            for n in s
                if isempty(∩(get_index_set(n.stats), union([get_index_set(n2.stats) for n2 in s if n2.node_id != n.node_id]...)))
                    is_cross_prod = true
                    break
                end
            end
            if !is_cross_prod
                push!(ss, s)
            end
        end
    end
    return ss
end

# This function takes in queries of the form:
#   Query(name, Aggregate(agg_op, idxs..., map_expr))
#   Query(name, Materialzie(formats.., idxs..., Aggregate(agg_op, idxs..., map_expr)))
# It outputs a set of queries where the final one binds `name` and each
# query has less than `MAX_INDEX_OCCURENCES` index occurences.
function split_query(ST, q::PlanNode)
    insert_node_ids!(q)
    aq = AnnotatedQuery(q, ST)
    pe = aq.point_expr
    insert_statistics!(ST, pe)
    has_agg = length(aq.reduce_idxs) > 0
    agg_op = has_agg ? aq.idx_op[first(aq.reduce_idxs)] : nothing
    node_id_counter = maximum([n.node_id for n in PostOrderDFS(pe)]) + 1
    queries = []
    remaining_idxs = aq.reduce_idxs
    cost_cache = Dict()
    cur_occurences = count_index_occurences([pe])
    while cur_occurences > MAX_INDEX_OCCURENCES
        nodes_to_remove = nothing
        new_query = nothing
        new_agg_idxs = nothing
        min_cost = Inf
        for node in PostOrderDFS(pe)
            if node.kind in (Value, Input, Alias, Index) || count_index_occurences([node]) == length(get_index_set(node.stats))
                continue
            end
            cache_key = [node.node_id]
            if !haskey(cost_cache, cache_key)
                    n_mat_stats =  has_agg ? node.stats : reduce_tensor_stats(agg_op.val, n_reduce_idxs, node.stats)
                    cost_cache[cache_key] = (get_reducible_idxs(aq, node),
                                            n_mat_stats,
                                            estimate_nnz(n_mat_stats))
            end
            n_reduce_idxs, n_mat_stats, n_cost = cost_cache[cache_key]
            if n_cost < min_cost && count_index_occurences([node]) < cur_occurences
                nodes_to_remove = [node.node_id]
                new_expr = (has_agg && !isempty(n_reduce_idxs)) ? Aggregate(agg_op, n_reduce_idxs..., node) : node
                new_expr.stats = n_mat_stats
                new_query = Query(Alias(gensym(:A)), new_expr)
                new_agg_idxs = n_reduce_idxs
                min_cost = n_cost
            end
            if node.kind == MapJoin && isassociative(node.op.val)
                for s in get_connected_subsets(node.args)
                    cache_key = sort([n.node_id for n in s])
                    if !haskey(cost_cache, cache_key)
                        s_stat = merge_tensor_stats(node.op.val, [n.stats for n in s]...)
                        s_reduce_idxs = Set{IndexExpr}()
                        for idx in n_reduce_idxs
                            if !any([idx ∈ get_index_set(n.stats) for n in setdiff(node.args, s)])
                                push!(s_reduce_idxs, idx)
                            end
                        end
                        s_mat_stats =  has_agg ? s_stat : reduce_tensor_stats(agg_op.val, s_reduce_idxs, s_stat)
                        s_cost = estimate_nnz(s_mat_stats) * AllocateCost
                        cost_cache[cache_key] = (s_reduce_idxs, s_mat_stats, s_cost)
                    end
                    s_reduce_idxs, s_mat_stats, s_cost = cost_cache[cache_key]
                    if s_cost < min_cost && count_index_occurences(s) > length(union([get_index_set(n.stats) for n in s]...))
                        nodes_to_remove = [n.node_id for n in s]
                        new_expr = (has_agg && !isempty(s_reduce_idxs)) ?
                                                Aggregate(agg_op, s_reduce_idxs..., MapJoin(node.op, s...)) :
                                                MapJoin(node.op, s...)
                        new_expr.stats = s_mat_stats
                        new_query = Query(Alias(gensym(:A)), new_expr)
                        new_agg_idxs = s_reduce_idxs
                        min_cost = s_cost
                    end
                end
            end
        end
        push!(queries, new_query)
        setdiff!(aq.reduce_idxs, new_agg_idxs)
        condense_stats!(new_query.expr.stats; cheap=false)
        alias = deepcopy(new_query.name)
        alias.stats = deepcopy(new_query.expr.stats)
        alias.node_id = node_id_counter
        node_id_counter += 1
        replace_and_remove_nodes!(pe, nodes_to_remove[1], alias, nodes_to_remove[2:end])
        cur_occurences = count_index_occurences([pe])
    end
    remainder_expr = has_agg ? Aggregate(agg_op, remaining_idxs..., pe) : pe

    final_query = Query(q.name, remainder_expr)
    push!(queries,  final_query)
    for query in queries
        insert_statistics!(ST, query)
    end
    return queries
end

function split_queries(ST, p::PlanNode)
    new_queries = []
    for query in p.queries
        append!(new_queries, split_query(ST, query))
    end
    new_plan = Plan(new_queries..., p.outputs)
    insert_node_ids!(new_plan)
    return new_plan
end
