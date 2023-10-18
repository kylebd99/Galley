# This file defines a query optimizer. It takes in both a query plan and input data, and it outputs an optimized query plan.
# It does this by gathering simple statistics about the data then doing a cost-based optimization based on equality saturation.

function EGraphs.make(::Val{:TensorStatsAnalysis}, g::EGraph, n::ENodeLiteral)
    if n.value isa Set
        return TensorStats(collect(n.value), Dict(), 0, nothing)
    elseif n.value isa Vector
        return TensorStats(n.value, Dict(), 0, nothing)
    else
        return  TensorStats([], Dict(), 0, n.value)
    end
end

# TODO: Replace these dictionaries with function definitions
annihilator_dict = Dict((*) => 0.0)
identity_dict = Dict((*) => 1.0, (+) => 0.0)

function merge_tensor_stats_join(op, lstats::TensorStats, rstats::TensorStats)
    new_default_value = op(lstats.default_value, rstats.default_value)
    new_indices = relative_sort(union(lstats.indices, rstats.indices), lstats.index_order)

    new_dim_size = Dict()
    for index in new_indices
        if index in lstats.indices && index in rstats.indices
            new_dim_size[index] = min(lstats.dim_size[index], rstats.dim_size[index])
        elseif index in rstats.indices
            new_dim_size[index] = rstats.dim_size[index]
        else
            new_dim_size[index] = lstats.dim_size[index]
        end
    end
    new_dim_space_size = prod([new_dim_size[x] for x in new_indices])
    l_dim_space_size = prod([lstats.dim_size[x] for x in lstats.indices])
    r_dim_space_size = prod([rstats.dim_size[x] for x in rstats.indices])
    l_prob_non_default = (lstats.cardinality/l_dim_space_size)
    r_prob_non_default = (rstats.cardinality/r_dim_space_size)
    new_cardinality = l_prob_non_default * r_prob_non_default * new_dim_space_size
    return TensorStats(new_indices, new_dim_size, new_cardinality, new_default_value, lstats.index_order)
end

function merge_tensor_stats_union(op, lstats::TensorStats, rstats::TensorStats)
    new_default_value = op(lstats.default_value, rstats.default_value)
    new_indices = relative_sort(union(lstats.indices, rstats.indices), rstats.index_order)

    new_dim_size = Dict()
    for index in new_indices
        if index in lstats.indices && index in rstats.indices
            new_dim_size[index] = max(lstats.dim_size[index], rstats.dim_size[index])
        elseif index in rstats.indices
            new_dim_size[index] = rstats.dim_size[index]
        else
            new_dim_size[index] = lstats.dim_size[index]
        end
    end

    new_dim_space_size = prod([new_dim_size[x] for x in new_indices])
    l_dim_space_size = prod([lstats.dim_size[x] for x in lstats.indices])
    r_dim_space_size = prod([rstats.dim_size[x] for x in rstats.indices])
    l_prob_default = (1 - lstats.cardinality/l_dim_space_size)
    r_prob_default = (1 - rstats.cardinality/r_dim_space_size)
    new_cardinality = (1 - l_prob_default * r_prob_default) * new_dim_space_size
    return TensorStats(new_indices, new_dim_size, new_cardinality, new_default_value, lstats.index_order)
end

function merge_tensor_stats(op, lstats::TensorStats, rstats::TensorStats)
    if !haskey(annihilator_dict, :($op))
        return merge_tensor_stats_union(op, lstats, rstats)
    end

    annihilator_value = annihilator_dict[:($op)]
    if annihilator_value == lstats.default_value && annihilator_value == rstats.default_value
        return merge_tensor_stats_join(op, lstats, rstats)
    else
        return merge_tensor_stats_union(op, lstats, rstats)
    end
end

function reduce_tensor_stats(op, reduce_indices::Vector{IndexExpr}, stats::TensorStats)
    indices = relative_sort(intersect(stats.indices, reduce_indices), stats.index_order)
    new_default_value = nothing
    if haskey(identity_dict, :($op)) && identity_dict[:($op)] == stats.default_value
        new_default_value = stats.default_value
    elseif op == +
        new_default_value = stats.default_value * prod([stats.dim_size[x] for x in indices])
    elseif op == *
        new_default_value = stats.default_value ^ prod([stats.dim_size[x] for x in indices])
    else
        # This is going to be VERY SLOW. Should raise a warning about reductions over non-identity default values.
        # Depending on the semantics of reductions, we might be able to do this faster.
        println("Warning: A reduction can take place over a tensor whose default value is not the reduction operator's identity. \\
                         This can result in a large slowdown as the new default is calculated.")
        new_default_value = op([stats.default_value for _ in prod([stats.dim_size[x] for x in indices])]...)
    end

    new_indices = relative_sort(setdiff(stats.indices, indices), stats.index_order)
    new_dim_size = Dict()
    for index in new_indices
        new_dim_size[index] = stats.dim_size[index]
    end


    new_dim_space_size = 1
    if length(new_indices) > 0
        new_dim_space_size = prod([new_dim_size[x] for x in new_indices])
    end
    old_dim_space_size = 1
    if length(stats.indices) > 0
        old_dim_space_size = prod([stats.dim_size[x] for x in stats.indices])
    end
    prob_default_value = 1 - stats.cardinality/old_dim_space_size
    prob_non_default_subspace = 1 - prob_default_value ^ (old_dim_space_size/new_dim_space_size)
    new_cardinality = new_dim_space_size * prob_non_default_subspace
    return TensorStats(new_indices, new_dim_size, new_cardinality, new_default_value, stats.index_order)
end

# This analysis function could support general statistics merging
function EGraphs.make(::Val{:TensorStatsAnalysis}, g::EGraph, n::ENodeTerm)
    if exprhead(n) == :call && operation(n) == MapJoin
        # Get the left and right child eclasses
        child_eclasses = arguments(n)
        op = g[child_eclasses[1]][1].value
        l = g[child_eclasses[2]]
        r = g[child_eclasses[3]]
        lstats = getdata(l, :TensorStatsAnalysis, nothing)
        rstats = getdata(r, :TensorStatsAnalysis, nothing)
        # If one of the arguments is a scalar, return the other argument's stats,
        # with the default value modified appropriately.
        if length(lstats.dim_size) == 0
            return TensorStats(rstats.indices, rstats.dim_size, rstats.cardinality, op(lstats.default_value, rstats.default_value), rstats.index_order)
        elseif length(rstats.dim_size) == 0
            return TensorStats(lstats.indices, lstats.dim_size, lstats.cardinality, op(lstats.default_value, rstats.default_value), lstats.index_order)
        else
            return merge_tensor_stats(op, lstats, rstats)
        end
    elseif exprhead(n) == :call && operation(n) == Aggregate
        op = operation(n)
        # Get the left and right child eclasses
        child_eclasses = arguments(n)
        op = g[child_eclasses[1]][1].value
        indices = g[child_eclasses[2]][1].value
        r = g[child_eclasses[3]]

        # Return the union of the index sets for MapJoin operators
        rstats = getdata(r, :TensorStatsAnalysis, nothing)
        return reduce_tensor_stats(op, indices, rstats)

    elseif exprhead(n) == :call && operation(n) == Reorder
        op = operation(n)
        # Get the left and right child eclasses
        child_eclasses = arguments(n)
        l = g[child_eclasses[1]]
        r = g[child_eclasses[2]]

        # Return the union of the index sets for MapJoin operators
        stats = getdata(l, :TensorStatsAnalysis, nothing)
        output_order = r[1].value
        sorted_indices = relative_sort(stats.indices, output_order)
        return TensorStats(sorted_indices, stats.dim_size,
                            stats.cardinality, stats.default_value, stats.index_order)
    elseif exprhead(n) == :call && operation(n) == RenameIndices
        op = operation(n)
        child_eclasses = arguments(n)
        stats = getdata(g[child_eclasses[1]], :TensorStatsAnalysis, nothing)
        output_indices = g[child_eclasses[2]][1].value
        return TensorStats(output_indices, Dict(x => 0 for x in output_indices),
                            stats.cardinality, stats.default_value, stats.index_order)
    elseif exprhead(n) == :call && operation(n) == InputTensor
        child_eclasses = arguments(n)
        indices::Vector{IndexExpr} = g[child_eclasses[1]][1].value
        fiber = g[child_eclasses[2]][1].value
        index_order::Vector{IndexExpr} = g[child_eclasses[3]][1].value
        return TensorStats(indices, fiber, index_order)
    elseif exprhead(n) == :call && operation(n) == Scalar
        return TensorStats([], Dict(), 1, n.args[1], [])
    end

    println("Warning! The following Tensor Kernel returned a `TensorStatsAnalysis` of `nothing`: ", n)
    println(exprhead(n))
    println(exprhead(n) == :call)
    println(typeof(operation(n)), "   ", typeof(MapJoin))
    println(operation(n), "   ", MapJoin)
    println(operation(n) == MapJoin)

    return nothing
end

EGraphs.islazy(::Val{:TensorStatsAnalysis})  = false

function EGraphs.join(::Val{:TensorStatsAnalysis}, a, b)
    if a.indices == b.indices && a.dim_size == b.dim_size && a.default_value == b.default_value
        return TensorStats(a.indices, a.dim_size, min(a.cardinality, b.cardinality), a.default_value, a.index_order)
    else
        println(a, "  ", b)
        println("EGraph Error: E-Nodes within an E-Class should never have different tensor types!")
        return nothing
    end
end

# A simple cost function which just denotes the cost of an E-Node as the sum
# of the incoming cardinalities.
function simple_cardinality_cost_function(n::ENodeTerm, g::EGraph)
    cost = 0
    for id in arguments(n)
        eclass = g[id]
        # if the child e-class has not yet been analyzed, return +Inf
        !hasdata(eclass, :TensorStatsAnalysis) && (cost += Inf; break)
        input_tensor_stats = getdata(eclass, :TensorStatsAnalysis)

        if !(input_tensor_stats === nothing)
            cost += input_tensor_stats.cardinality
        end
    end
    return cost
end

# All literal expressions have cost 0
simple_cardinality_cost_function(n::ENodeLiteral, g::EGraph) = 0

doesnt_share_indices(is, a) = false

function doesnt_share_indices(is::Vector{IndexExpr}, a::EClass)
    return length(intersect(is, getdata(a, :TensorStatsAnalysis, nothing).indices)) == 0
end

function doesnt_share_indices(a::EClass, b::EClass)
    if length(getdata(a, :TensorStatsAnalysis, nothing).indices) == 0 || length(getdata(b, :TensorStatsAnalysis, nothing).indices) == 0
        return false
    end
    return length(intersect(getdata(a, :TensorStatsAnalysis, nothing).indices, getdata(b, :TensorStatsAnalysis, nothing).indices)) == 0
end


# This theory has the 6/7 of the RA rules from SPORES. Currently, it's missing the
# reduction removal one because it requires knowing the dimensionality which isn't available in the local context.
# Additionally, it doesn't allow cross-products to reduce the search space.
# TODO:
# - Automatically produce this list of rewrites based on the operators present in the
#      query and their properties. Some operators only become present after a transformation
#      such as x*x => x^2, so we will need to have some kind of transitive closure over the
#      rules.
# - Find a way to express the full reduction when the inner expression doesn't include
#    the reducing variables.
basic_rewrites = @theory a b c d f is js begin
    # Fuse reductions
    Aggregate(f, is, Aggregate(f, js, a)) => Aggregate(f, Vector{IndexExpr}(collect(Set(union(is, js)))), a)

    # UnFuse reductions
    Aggregate(f, is, a) => Aggregate(f, is[1:1], Aggregate(f, Vector{IndexExpr}(is[2:length(is)]), a)) where (length(is) > 1)

    # Reorder Reductions
    Aggregate(f, is, Aggregate(f, js, a)) == Aggregate(f, js, Aggregate(f, is, a))

    # Commutativity
    MapJoin(+, a, b) == MapJoin(+, b, a)
    MapJoin(*, a, b) == MapJoin(*, b, a)
    Aggregate(+, is, MapJoin(+, a, b)) == MapJoin(+, Aggregate(+, is, a), Aggregate(+, is, b))

    # Associativity
    MapJoin(+, a, MapJoin(+, b, c)) =>  MapJoin(+, MapJoin(+, a, b), c) where (!doesnt_share_indices(b, a))
    MapJoin(*, a, MapJoin(*, b, c)) => MapJoin(*, MapJoin(*, a, b), c) where (!doesnt_share_indices(b, a))

    # Distributivity
    MapJoin(*, a, MapJoin(+, b, c)) == MapJoin(+, MapJoin(*, a, b), MapJoin(*, a, c))

    # Reduction PushUp
    MapJoin(*, a, Aggregate(+, is, b)) => Aggregate(+, is, MapJoin(*, a, b)) where (doesnt_share_indices(is, a))

    # Reduction PushDown
    Aggregate(+, is, MapJoin(*, a, b)) => MapJoin(*, a, Aggregate(+, is, b)) where (doesnt_share_indices(is, a))

    # Handling Squares
    MapJoin(^, a, Scalar(2)) == MapJoin(*, a, a)
    MapJoin(^, MapJoin(^, a, c), d) == MapJoin(^, a, MapJoin(*, c, d))

    # Handling Simple Duplicates
    MapJoin(+, a, a) == MapJoin(*, a, 2)

    # Reduction removal, need a dimension->size dict
    #Aggregate(+, is::Set, a) => :(MapJoin(*, a, dim(is)))
end

# These functions unwrap an EGraph object which is returned from saturation into a
# logical plan expression.
function e_graph_to_expr_tree(g::EGraph, index_order)
    return e_class_to_expr_node(g, g[g.root], index_order)
end

function e_class_to_expr_node(g::EGraph, e::EClass, index_order; verbose=0)
    n = e[1]
    stats = getdata(e, :TensorStatsAnalysis)
    children = []
    if n isa ENodeTerm
        for c in arguments(n)
            if g[c][1] isa ENodeTerm
                push!(children, e_class_to_expr_node(g, g[c], index_order))
            elseif g[c][1].value isa Number && operation(n) != Scalar
                push!(children, Scalar(g[c][1].value, getdata(g[c], :TensorStatsAnalysis)))
            else
                push!(children, g[c][1].value)
            end
        end
    end

    return LogicalPlanNode(operation(n), children, stats)
end
