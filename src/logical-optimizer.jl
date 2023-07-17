# This file defines a query optimizer. It takes in both a query plan and input data, and it outputs an optimized query plan.
# It does this by gathering simple statistics about the data then doing a cost-based optimization based on equality saturation.
using Metatheory
using Metatheory.EGraphs
using PrettyPrinting
using AutoHashEquals
include("logical-query-plan.jl")


function relativeSort(indices::Vector{String}, index_order; rev=false)
    if index_order === nothing
        return indices
    end
    sorted_indices::Vector{String} = []
    if rev == false
        for idx in index_order
            if idx in indices
                push!(sorted_indices, idx)
            end
        end
        return sorted_indices
    else
        for idx in reverse(index_order)
            if idx in indices
                push!(sorted_indices, idx)
            end
        end
        return sorted_indices
    end
end

function isSortedWRTIndexOrder(indices::Vector, index_order::Vector)
    return issorted(indexin(indices, index_order))
end

function addGlobalOrder(x::LogicalPlanNode, global_index_order)
    return LogicalPlanNode(x.head, [x.args..., global_index_order], nothing)
end

function insertGlobalOrders(expr, global_index_order)
    global_order_rule = @rule ~x => addGlobalOrder(x, global_index_order) where (x isa LogicalPlanNode && x.head == InputTensor)
    global_order_rule = Metatheory.Postwalk(Metatheory.PassThrough(global_order_rule))
    return global_order_rule(expr)
end

needsReorder(expr, index_order) = false
function needsReorder(expr::LogicalPlanNode, index_order)
    return expr.head == InputTensor && !isSortedWRTIndexOrder(expr.args[1], index_order)
end

function insertInputReorders(expr, global_index_order)
    reorder_rule = @rule ~x => Reorder(x, global_index_order) where (needsReorder(x, global_index_order))
    reorder_rule = Metatheory.Postwalk(Metatheory.PassThrough(reorder_rule))
    return reorder_rule(expr)
end

function removeUnecessaryReorders(expr, global_index_order)
    if expr.head == Reorder && isSortedWRTIndexOrder(expr.args[2], global_index_order)
        return expr.args[1]
    end
    return expr
end

function mergeAggregates(expr)
    merge_rule = @rule op idx_1 idx_2 x  Aggregate(op, idx_1, Aggregate(op, idx_2, x)) => Aggregate(op, union(idx_1, idx_2), x)
    merge_rule = Metatheory.Postwalk(Metatheory.PassThrough(merge_rule))
    new_expr = merge_rule(expr)
    if new_expr === nothing
        return expr
    else
        return new_expr
    end
end


function recursive_rename(expr::LogicalPlanNode, index_lookup, depth, context, context_counter, drop_stats, drop_index_order)
    if expr.head == RenameIndices
        expr_index_lookup = Dict()
        renamed_indices::Vector{String} = expr.args[2]
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
        indices = Vector{String}()
        for index in expr.stats.indices
            if index in keys(index_lookup)
                push!(indices, index_lookup[index])
            elseif context > 0
                push!(indices, index * "_" * string(context))
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
        elseif arg isa Vector{String}
            new_indices = Vector{String}()
            for index in arg
                if index in keys(index_lookup)
                    push!(new_indices, index_lookup[index])
                elseif context > 0
                    push!(new_indices, index * "_" * string(context))
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




function EGraphs.make(::Val{:TensorStatsAnalysis}, g::EGraph, n::ENodeLiteral)
    if n.value isa Set
        return TensorStats(collect(n.value), Dict(), 0, nothing)
    elseif n.value isa Vector
        return TensorStats(n.value, Dict(), 0, nothing)
    else
        return  TensorStats([], Dict(), 0, n.value)
    end
end

annihilator_dict = Dict((*) => 0.0)
identity_dict = Dict((*) => 1.0, (+) => 0.0)

function mergeTensorStatsJoin(op, lstats::TensorStats, rstats::TensorStats)
    new_default_value = op(lstats.default_value, rstats.default_value)
    new_indices = relativeSort(union(lstats.indices, rstats.indices), lstats.index_order)

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

function mergeTensorStatsUnion(op, lstats::TensorStats, rstats::TensorStats)
    new_default_value = op(lstats.default_value, rstats.default_value)
    new_indices = relativeSort(union(lstats.indices, rstats.indices), rstats.index_order)

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


function mergeTensorStats(op, lstats::TensorStats, rstats::TensorStats)
    if !haskey(annihilator_dict, :($op))
        return mergeTensorStatsUnion(op, lstats, rstats)
    end

    annihilator_value = annihilator_dict[:($op)]
    if annihilator_value == lstats.default_value && annihilator_value == rstats.default_value
        return mergeTensorStatsJoin(op, lstats, rstats)
    else
        return mergeTensorStatsUnion(op, lstats, rstats)
    end
end

function reduceTensorStats(op, indexStats, stats::TensorStats)
    indices = relativeSort(intersect(stats.indices, indexStats.indices), stats.index_order)
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

    new_indices = relativeSort(setdiff(stats.indices, indices), stats.index_order)
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
            return mergeTensorStats(op, lstats, rstats)
        end
    elseif exprhead(n) == :call && operation(n) == Aggregate
        op = operation(n)
        # Get the left and right child eclasses
        child_eclasses = arguments(n)
        op = g[child_eclasses[1]][1].value
        l = g[child_eclasses[2]]
        r = g[child_eclasses[3]]

        # Return the union of the index sets for MapJoin operators
        indices = getdata(l, :TensorStatsAnalysis, nothing)
        rstats = getdata(r, :TensorStatsAnalysis, nothing)
        return reduceTensorStats(op, indices, rstats)

    elseif exprhead(n) == :call && operation(n) == Reorder
        op = operation(n)
        # Get the left and right child eclasses
        child_eclasses = arguments(n)
        l = g[child_eclasses[1]]
        r = g[child_eclasses[2]]

        # Return the union of the index sets for MapJoin operators
        stats = getdata(l, :TensorStatsAnalysis, nothing)
        output_order = r[1].value
        sorted_indices = relativeSort(stats.indices, output_order)
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
        indices = g[child_eclasses[1]][1].value
        fiber = g[child_eclasses[2]][1].value
        index_order = g[child_eclasses[3]][1].value
        return TensorStats(indices, fiber, index_order)
    elseif exprhead(n) == :call && operation(n) == Scalar
        return TensorStats([], Dict(), 1, n.args[1], [])
    end

    println("Warning! The following Tensor Kernel returned a `TensorStatsAnalysis` of `nothing`: ", n)
    println(exprhead(n))
    println(exprhead(n) == :call)
    println(typeof(operation(n)), "   ", typeof(:MapJoin))
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

doesntShareIndices(is, a) = false

function doesntShareIndices(is::Vector, a::EClass)
    return length(intersect(is, getdata(a, :TensorStatsAnalysis, nothing).indices)) == 0
end

function doesntShareIndices(a::EClass, b::EClass)
    if length(getdata(a, :TensorStatsAnalysis, nothing).indices) == 0 || length(getdata(b, :TensorStatsAnalysis, nothing).indices) == 0
        return false
    end
    return length(intersect(getdata(a, :TensorStatsAnalysis, nothing).indices, getdata(b, :TensorStatsAnalysis, nothing).indices)) == 0
end


# This theory has the 6/7 of the RA rules from SPORES. Currently, it's missing the
# reduction removal one because it requires knowing the dimensionality which isn't available in the local context.
# Additionally, it doesn't allow cross-products to reduce the search space.
basic_rewrites = @theory a b c d f is js begin
    # Fuse reductions
    Aggregate(f, is, Aggregate(f, js, a)) => Aggregate(f, union(is, js), a)

    # UnFuse reductions
    Aggregate(f, is, a) => Aggregate(f, is[1:1], Aggregate(f, is[2:length(is)], a)) where (length(is) > 1)

    # Reorder Reductions
    Aggregate(f, is, Aggregate(f, js, a)) == Aggregate(f, js, Aggregate(f, is, a))

    # Commutativity
    MapJoin(+, a, b) == MapJoin(+, b, a)
    MapJoin(*, a, b) == MapJoin(*, b, a)
    Aggregate(+, is, MapJoin(+, a, b)) == MapJoin(+, Aggregate(+, is, a), Aggregate(+, is, b))

    # Associativity
    MapJoin(+, a, MapJoin(+, b, c)) =>  MapJoin(+, MapJoin(+, a, b), c) where (!doesntShareIndices(b, a))
    MapJoin(*, a, MapJoin(*, b, c)) => MapJoin(*, MapJoin(*, a, b), c) where (!doesntShareIndices(b, a))

    # Distributivity
    MapJoin(*, a, MapJoin(+, b, c)) == MapJoin(+, MapJoin(*, a, b), MapJoin(*, a, c))

    # Reduction PushUp
    MapJoin(*, a, Aggregate(+, is::Set, b)) => Aggregate(+, is, MapJoin(*, a, b)) where (doesntShareIndices(is, a))

    # Reduction PushDown
    Aggregate(+, is, MapJoin(*, a, b)) => MapJoin(*, a, Aggregate(+, is, b)) where (doesntShareIndices(is, a))

    # Handling Squares
    MapJoin(^, a, Scalar(2)) == MapJoin(*, a, a)
    MapJoin(^, MapJoin(^, a, c), d) == MapJoin(^, a, MapJoin(*, c, d))

    # Handling Simple Duplicates
    MapJoin(+, a, a) == MapJoin(*, a, 2)

    # Reduction removal, need a dimension->size dict
    #Aggregate(+, is::Set, a) => :(MapJoin(*, a, dim(is)))
end

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
