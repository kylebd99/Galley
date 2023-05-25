# This file defines a query optimizer. It takes in both a query plan and input data, and it outputs an optimized query plan.
# It does this by gathering simple statistics about the data then doing a cost-based optimization based on equality saturation. 
using Metatheory
using Metatheory.EGraphs
using TermInterface
using PrettyPrinting
include("LogicalQueryPlan.jl")

function EGraphs.isequal(x::TensorStats, y::TensorStats)
    if x.indices == y.indices && x.dim_size==y.dim_size && x.default_value == y.default_value
        return true
    else 
        return false
    end
end


function EGraphs.make(::Val{:TensorStatsAnalysis}, g::EGraph, n::ENodeLiteral)    
    if n.value isa InputTensor
        return n.value.stats
    elseif n.value isa Set 
        return TensorStats(n.value, Dict(), 0, nothing)
    else 
        return  TensorStats(Set(), Dict(), 0, n.value)
    end
end

annihilator_dict = Dict((*) => 0.0)
identity_dict = Dict((*) => 1.0, (+) => 0.0)

function mergeTensorStatsJoin(op, lstats::TensorStats, rstats::TensorStats)
    new_default_value = op(lstats.default_value, rstats.default_value)
    new_indices = union(lstats.indices, rstats.indices)

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
#    println("Left Cardinality: ", lstats.cardinality)
#    println("Right Cardinality: ", rstats.cardinality)
#    println("Cardinality: ", new_cardinality)

    return TensorStats(new_indices, new_dim_size, new_cardinality, new_default_value)
end

function mergeTensorStatsUnion(op, lstats::TensorStats, rstats::TensorStats)
    new_default_value = op(lstats.default_value, rstats.default_value)
    new_indices = union(lstats.indices, rstats.indices)

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

    return TensorStats(new_indices, new_dim_size, new_cardinality, new_default_value)
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
    indices = intersect(stats.indices, indexStats.indices)
    new_default_value = nothing
    if identity_dict[:($op)] == stats.default_value
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

    new_indices = setdiff(stats.indices, indices)
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
#    println("Prob Default Value: ", prob_default_value)
#    println("Previous Cardinality: ", stats.cardinality)
#    println("Cardinality: ", new_cardinality)
    return TensorStats(new_indices, new_dim_size, new_cardinality, new_default_value)
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
            return TensorStats(rstats.indices, rstats.dim_size, rstats.cardinality, op(lstats.default_value, rstats.default_value))
        elseif length(rstats.dim_size) == 0
            return TensorStats(lstats.indices, lstats.dim_size, lstats.cardinality, op(lstats.default_value, rstats.default_value))
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
    end
    println("Warning! The following Tensor Kernel returned a `TensorStatsAnalysis` of `nothing`: ", n)
    return nothing
end

EGraphs.islazy(::Val{:TensorStatsAnalysis})  = false

function EGraphs.join(::Val{:TensorStatsAnalysis}, a, b)
    if a.indices == b.indices && a.dim_size == b.dim_size && a.default_value == b.default_value
        return TensorStats(a.indices, a.dim_size, min(a.cardinality, b.cardinality), a.default_value)
    else
        println(a, "  ", b)
        println("EGraph Error: E-Nodes within an E-Class should never have different tensor types!")
        # an expression cannot be odd and even at the same time!
        # this is contradictory, so we ignore the analysis value
        return nothing 
    end
end

# This is a cost function that behaves like `astsize` but increments the cost 
# of nodes containing the `^` operation. This results in a tendency to avoid 
# extraction of expressions containing '^'.
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

function ReduceDim end
function MapJoin end
function mapScalar end

doesntShareIndices(is, a) = false

function doesntShareIndices(is::Set, a::EClass)
    return length(intersect(is, getdata(a, :TensorStatsAnalysis, nothing).indices)) == 0
end

function doesntShareIndices(is::Set, a::InputTensor) 
    return length(intersect(is, a.stats.indices)) == 0
end

function doesntShareIndices(a::EClass, b::EClass) 
    return length(intersect(getdata(a, :TensorStatsAnalysis, nothing).indices, getdata(b, :TensorStatsAnalysis, nothing).indices)) == 0
end


# This theory has the 6/7 of the RA rules from SPORES. Currently, it's missing the 
# reduction removal one because it requires knowing the dimensionality which isn't available in the local context.
# Additionally, it doesn't allow cross-products to reduce the search space.
basic_rewrites = @theory a b c d f is js begin
    # Fuse reductions
    ReduceDim(f, is::Set, ReduceDim(f, js::Set, a)) => :(ReduceDim($f, $(union(is, js)), $a))

    # UnFuse reductions
    ReduceDim(f, is::Set, a) => :(ReduceDim($f, $(Set([first(is)])), ReduceDim($f, $(setdiff(is, Set([first(is)]))), $a))) where (length(is) > 1)

    # Reorder Reductions
    ReduceDim(f, is::Set, ReduceDim(f, js::Set, a)) == ReduceDim(f, js::Set, ReduceDim(f, is::Set, a))

    # Commutativity
    MapJoin(+, a, b) == MapJoin(+, b, a)
    MapJoin(*, a, b) == MapJoin(*, b, a)
    ReduceDim(+, is::Set, MapJoin(+, a, b)) == MapJoin(+, ReduceDim(+, is, a), ReduceDim(+, is, b))

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

    # Reduction removal, need a dimension->size dict
    #ReduceDim(+, is::Set, a) => :(MapJoin(*, a, dim(is)))
    #ReduceDim(+, is::Set, MapJoin(*, a, b)) => :(ReduceDim($+,  $(intersect(is, getdata(a, :IndexAnalysis, nothing))), MapJoin($*, $a, ReduceDim($+, $(setdiff(is, getdata(a, :IndexAnalysis, nothing))), $b))))
end