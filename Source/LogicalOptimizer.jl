# This file defines a query optimizer. It takes in both a query plan and input data, and it outputs an optimized query plan.
# It does this by gathering simple statistics about the data then doing a cost-based optimization based on equality saturation. 
using Metatheory
using Metatheory.EGraphs
using TermInterface
using PrettyPrinting
using AutoHashEquals
include("LogicalQueryPlan.jl")


function relativeSort(indices, index_order; rev=false)
    if index_order === nothing
        return indices
    end
    if rev == false
        sorted_indices = []
        for idx in index_order
            if idx in indices
                push!(sorted_indices, idx)
            end
        end
        return sorted_indices
    else
        sorted_indices = []
        for idx in reverse(index_order)
            if idx in indices
                push!(sorted_indices, idx)
            end
        end
        return sorted_indices

    end
end

function isSortedWRTIndexOrder(indices::Vector{String}, index_order::Vector)
    return issorted(indexin(indices, index_order))
end

function needsReorder(expr, index_order)
    return expr isa InputTensor && !isSortedWRTIndexOrder(expr.stats.indices, index_order)
end

function needsGlobalOrder(expr)
    return expr isa InputTensor
end

function insertGlobalOrders(expr, global_index_order)
    global_order_rule = @rule ~x => :($(InputTensor(x.tensor_id, x.fiber, TensorStats(x.stats.indices, x.stats.dim_size, x.stats.cardinality, x.stats.default_value, global_index_order)))) where needsGlobalOrder(x)
    global_order_rule = Metatheory.Postwalk(Metatheory.PassThrough(global_order_rule))
    return global_order_rule(expr)
end

function insertInputReorders(expr, global_index_order)
    reorder_rule = @rule ~x => :(Reorder($x, $global_index_order)) where needsReorder(x, global_index_order)
    reorder_rule = Metatheory.Postwalk(Metatheory.PassThrough(reorder_rule))
    return reorder_rule(expr)
end


function removeUnecessaryReorders(expr)
    remove_reorder_rule = @rule Reorder(~x, ~index_order) => ~x where !needsReorder(x, index_order)
    remove_reorder_rule = Metatheory.Postwalk(Metatheory.PassThrough(remove_reorder_rule))
    return remove_reorder_rule(expr)
end

function mergeAggregates(expr)
    merge_rule = @rule op idx_1 idx_2 x  ReduceDim(op, idx_1, ReduceDim(op, idx_2, x)) => :(ReduceDim($op, $(union(idx_1, idx_2)), $x))
    merge_rule = Metatheory.Postwalk(Metatheory.PassThrough(merge_rule))
    new_expr = merge_rule(expr)
    if new_expr === nothing
        return expr
    else
        return new_expr
    end
end

#function EGraphs.isequal(x::TensorStats, y::TensorStats)
#    if x.indices == y.indices && x.dim_size==y.dim_size && x.default_value == y.default_value
#        return true
#    else 
#        return false
#    end
#end


function EGraphs.make(::Val{:TensorStatsAnalysis}, g::EGraph, n::ENodeLiteral)    
    if n.value isa InputTensor
        return n.value.stats
    elseif n.value isa Set 
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

    if exprhead(n) == :call && operation(n) == :MapJoin
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
    elseif exprhead(n) == :call && operation(n) == :ReduceDim
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
    
    elseif exprhead(n) == :call && operation(n) == :Reorder
        op = operation(n)
        # Get the left and right child eclasses
        child_eclasses = arguments(n)
        l = g[child_eclasses[1]]
        r = g[child_eclasses[2]]

        # Return the union of the index sets for MapJoin operators
        index_order = getdata(r, :TensorStatsAnalysis, nothing).indices
        stats = getdata(l, :TensorStatsAnalysis, nothing)
        sorted_indices = relativeSort(stats.indices, index_order)
        return TensorStats(sorted_indices, stats.dim_size, 
                            stats.cardinality, stats.default_value, index_order)
    end
    println("Warning! The following Tensor Kernel returned a `TensorStatsAnalysis` of `nothing`: ", n)
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
#    println(n, " Cost: ", cost)
    return cost
end

# All literal expressions (e.g `a`, 123, 0.42, "hello") have cost 1
simple_cardinality_cost_function(n::ENodeLiteral, g::EGraph) = 0

doesntShareIndices(is, a) = false

function doesntShareIndices(is::Vector, a::EClass)
    return length(intersect(is, getdata(a, :TensorStatsAnalysis, nothing).indices)) == 0 
end

function doesntShareIndices(is::Vector, a::InputTensor) 
    return length(intersect(is, a.stats.indices)) == 0
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
    ReduceDim(f, is, ReduceDim(f, js, a)) => :(ReduceDim($f, $(union(is, js)), $a))

    # UnFuse reductions
    ReduceDim(f, is, a) => :(ReduceDim($f, $([first(is)]), ReduceDim($f, $(setdiff(is, [first(is)])), $a))) where (length(is) > 1)

    # Reorder Reductions
    ReduceDim(f, is, ReduceDim(f, js, a)) == ReduceDim(f, js, ReduceDim(f, is, a))

    # Commutativity
    MapJoin(+, a, b) == MapJoin(+, b, a)
    MapJoin(*, a, b) == MapJoin(*, b, a)
    ReduceDim(+, is, MapJoin(+, a, b)) == MapJoin(+, ReduceDim(+, is, a), ReduceDim(+, is, b))

    # Associativity
    MapJoin(+, a, MapJoin(+, b, c)) =>  :(MapJoin($+, MapJoin($+, $a, $b), $c)) where (!doesntShareIndices(b, a))
    MapJoin(*, a, MapJoin(*, b, c)) => :(MapJoin($*, MapJoin($*, $a, $b), $c)) where (!doesntShareIndices(b, a))
    
    # Distributivity
    MapJoin(*, a, MapJoin(+, b, c)) == MapJoin(+, MapJoin(*, a, b), MapJoin(*, a, c))

    # Reduction PushUp
    MapJoin(*, a, ReduceDim(+, is::Set, b)) => :(ReduceDim($+, $is, MapJoin($*, $a, $b))) where (doesntShareIndices(is, a))

    # Reduction PushDown
    ReduceDim(+, is, MapJoin(*, a, b)) => :(MapJoin($*, $a, ReduceDim($+, $is, $b))) where (doesntShareIndices(is, a))

    # Handling Squares
    MapJoin(^, a, 2) == MapJoin(*, a, a)
    MapJoin(^, MapJoin(^, a, c), d) => :(MapJoin($^, $a, $(c*d))) where (c isa Number && d isa Number)

    # Handling Simple Duplicates
    MapJoin(+, a, a) == MapJoin(*, a, 2)

    Reorder(~x, ~index_order) => ~x where IsSortedWRTIndexOrder(getdata(x, :TensorStatsAnalysis, nothing).indices, index_order)

    # Reduction removal, need a dimension->size dict
    #ReduceDim(+, is::Set, a) => :(MapJoin(*, a, dim(is)))
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
            elseif g[c][1].value isa InputTensor
                push!(children, g[c][1].value)
            elseif g[c][1].value isa Number
                push!(children, Scalar(g[c][1].value, getdata(g[c], :TensorStatsAnalysis)))            
            else
                push!(children, g[c][1].value)
            end
        end
    end
    nodeType = eval(operation(n))
    return nodeType(children..., stats, nothing)
end

function label_expr_parents!(parent, cur_node::LogicalPlanNode)
    cur_node.parent = parent
    if cur_node isa ReduceDim && cur_node.input isa LogicalPlanNode
            label_expr_parents!(cur_node, cur_node.input)
    elseif cur_node isa Reorder && cur_node.input isa LogicalPlanNode
            label_expr_parents!(cur_node, cur_node.input)
    elseif cur_node isa MapJoin
        if cur_node.left_input isa LogicalPlanNode
            label_expr_parents!(cur_node, cur_node.left_input)
        end
        if cur_node.right_input isa LogicalPlanNode
            label_expr_parents!(cur_node, cur_node.right_input)
        end
    end
    return cur_node
end