# This optimizer is based off of the approach from
# "Constructing Optimal Contraction Trees for Tensor Network Quantum Circuit Simulation"
# (https://ieeexplore.ieee.org/document/9926353)
# It splits the optimization process into a heuristic NP-Hard part and a deterministic
# quadratic part. 


function generate_orders(factors::Set{Factor})
    return [collect(factors)]
end

function get_sub_order_cost!(sum_op,
                            mult_op,
                            output_indices::Set{IndexExpr},
                            factors::Vector{Factor},
                            index_ranges::Dict{IndexExpr, Tuple{Int, Int}},
                            stats::Dict{Tuple{Int, Int}, TensorStats},
                            costs::Dict{Tuple{Int, Int}, Float64},
                            optimal_subtrees::Dict{Tuple{Int, Int}, Union{Bag, Factor}},
                            i::Int,
                            j::Int)
    # Base case where the sub-order is a single tensor
    if i == j
        costs[(i,j)] = estimate_nnz(stats[(i,j)])
        optimal_subtrees[(i,j)] = factors[i]
    end

    min_cost = Inf
    for k in i : 1 : j-1
        max_k_cost = 0
        if !haskey(costs, (i, k))
            get_sub_order_cost(sum_op, mult_op, factors, index_ranges, stats, costs, optimal_subtrees, i, k)
            max_k_cost = max(max_k_cost, costs[(i, k)])
        end

        if !haskey(costs, (k+1, j))
            get_sub_order_cost(sum_op, mult_op, factors, index_ranges, stats, costs, optimal_subtrees, k+1, j)
            max_k_cost = max(max_k_cost, costs[(k+1, j)])
        end

        join_indices = ∪(get_index_set(stats[(i,k)]), get_index_set(stats[(k+1,j)]))
        aggregated_indices = Set()
        for index in setdiff(join_indices, output_indices)
            range = index_ranges[index]
            if range[1] >= i && range[2] <= j
                push!(aggregated_indices, index)
            end
        end

        if !haskey(stats, (i,j))
            join_stats = merge_tensor_def_join(mult_op, stats[(i,k)], stats[(k+1,j)])
            reduce_stats = if length(aggregated_indices) > 0
                reduce_tensor_stats(sum_op, aggregate_indices, join_stats)
            else
                join_stats
            end
            stats[(i,j)] = reduce_stats
        end

        max_k_cost = max(max_k_cost, estimate_nnz(stats[(i,j)]))
        if min_cost > max_k_cost
            min_cost = max_k_cost
            edge_cover = Set{Factor}()
            child_bags = Set{Bag}()
            if i == k
                push!(edge_cover, factors[i])
            else
                push!(child_bags, optimal_subtrees[(i,k)])
            end

            if k+1==j
                push!(edge_cover, factors[j])
            else
                push!(child_bags, optimal_subtrees[(k+1, j)])
            end

            stat = stats[(i,j)]
            covered_indices = ∪(get_index_set(stats[(i,k)]), get_index_set(stats[(k+1,j)]))
            parent_indices = aggregated_indices
            optimal_subtrees[(i,j)]= Bag(edge_cover, covered_indices, parent_indices, child_bags, stat)
        end
    end
end


function optimal_htd_given_order(sum_op, mult_op, factors::Vector{Factor}, output_indices::Set{IndexExpr}, output_index_order::Vector{IndexExpr})
    indices = ∪([f.all_indices for f in factors]...)
    index_ranges = Dict{IndexExpr, Tuple{Int, Int}}()
    for index in indices
        l = length(factors)
        r = 1
        for i in eachindex(factors)
            if index in factors[i].all_indices
                l = min(l, i)
                r = max(r, i)
            end
        end
        index_ranges[index] = (l, r)
    end
    stats = Dict{Tuple{Int, Int}, TensorStats}()
    for i in eachindex(factors)
        stats[(i,i)] = factors[i].stats
    end
    costs = Dict{Tuple{Int, Int}, Float64}()
    optimal_subtrees = Dict{Tuple{Int, Int}, Union{Bag, Factor}}()

    get_sub_order_cost!(sum_op,
                        mult_op,
                        factors,
                        index_ranges,
                        stats,
                        costs,
                        optimal_subtrees,
                        1,
                        length(factors),
                        output_indices)

    if optimal_subtrees[(1, length(factors))] isa Factor
        factor = optimal_subtrees[(1, length(factors))]
        optimal_subtrees[(1, length(factors))] = Bag([factor],
                                                        factor.all_indices,
                                                        output_indices,
                                                        [],
                                                        factor.stats)
    end
    optimal_htd = HyperTreeDecomposition(mult_op,
                                sum_op,
                                output_indices,
                                optimal_subtrees[(1, length(factors))],
                                output_index_order)
    return (costs[1, length(factors)], optimal_htd)
end

function order_based_decomposition(faq::FAQInstance; verbose=0)
    orders = generate_orders(faq.factors)
    min_cost = Inf
    optimal_htd = nothing
    for order in orders
        cost, htd = optimal_htd_given_order(faq.sum_op,
                                            faq.mult_op,
                                            order,
                                            faq.output_indices,
                                            faq.output_index_order)
        if cost < min_cost
            min_cost = cost
            optimal_htd = htd
        end
    end

    return optimal_htd
end
