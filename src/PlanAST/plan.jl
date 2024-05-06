# This file defines the logical query plan (LQP) language.
# Each LQP is a tree of expressions where interior nodes refer to
# function calls and leaf nodes refer to constants or input tensors.
const IS_TREE = 1
const IS_STATEFUL = 2
const ID = 4

@enum PlanNodeKind begin
    Value        =  1ID             #  Value(x::Any)
    Index        =  2ID             #  Index(x::Union{String, Symbol})
    Alias        =  3ID             #  Alias(x::Union{String, Symbol})
    Input        =  4ID | IS_TREE   #  Input(tns::Union{Tensor, Number}, idxs...::{TI})
    MapJoin      =  5ID | IS_TREE   #  MapJoin(op::Value, args..::PlanNode)
    Aggregate    =  6ID | IS_TREE   #  Aggregate(op::Value, idxs...::Index, arg::PlanNode)
    Materialize  =  7ID | IS_TREE   #  Materialize(formats::Vector{Formats}, idx_order::Vector{TI}, arg:PlanNode)
    Query        =  8ID | IS_TREE   #  Query(name::Alias, expr::PlanNode)
    Outputs      =  9ID | IS_TREE   #  Outputs(args...::TI)
    Plan         = 10ID | IS_TREE   #  Alias(x::Union{String, Symbol})
end

# Here, we define the internal expression type that we use to describe logical query plans.
mutable struct PlanNode
    kind::PlanNodeKind
    children::Vector{PlanNode}
    val::Any
    stats::Any
    node_id::Int
end
PN = PlanNode

function PlanNode(kind::PlanNodeKind, children::Vector, val::Any, stats::Any)
    return PlanNode(kind, children, val, stats, -1)
end

function PlanNode(kind::PlanNodeKind, args::Vector)
    if (kind === Value || kind === Index || kind === Alias) && length(args) == 1
        return PlanNode(kind, PlanNode[], args[1], nothing)
    else
        args = vcat(args...)
        if (kind === Input && length(args) >= 1)
            if args[1] isa Tensor
                PlanNode(kind, args, nothing, nothing)
            elseif args[1].kind === Value
                PlanNode(kind, args, nothing, nothing)
            elseif args[1].kind === Materialize
                mat_expr = args[1]
                new_idxs = [Index(x) for x  in args[2:end]]
                old_idxs = [idx for idx in mat_expr.idx_order]
                @assert length(new_idxs) == length(old_idxs)
                idx_translate = merge(Dict(new_idxs[i].name => Index(gensym(new_idxs[i].name)) for i in eachindex(old_idxs)),
                                        Dict(old_idxs[i].name => new_idxs[i] for i in eachindex(old_idxs)))
                internal_expr = mat_expr.expr
                result_expr = Rewrite(Postwalk(@rule ~x => idx_translate[x.name] where  ((x.kind === Index) && (x.name in keys(idx_translate)))))(internal_expr)
                return result_expr
            else
                error("a reused plan expression must be wrapped in a materialize!")
            end
        elseif (kind === MapJoin && length(args) >= 2) ||
            (kind === Aggregate && length(args) >= 2) ||
            (kind === Materialize && length(args) >= 1 && length(args) % 2 == 1) ||
            (kind === Query && length(args) >= 2) ||
            (kind === Outputs) ||
            (kind === Plan)
            return PlanNode(kind, args, nothing, nothing)
        else
            error("wrong number of arguments to $kind(...)")
        end
    end
end

function (kind::PlanNodeKind)(args...)
    PlanNode(kind, Any[args...,])
end

# SyntaxInterface mandatories.
isvalue(node::PlanNode) = node.kind === Value
isindex(node::PlanNode) = node.kind === Index
isalias(node::PlanNode) = node.kind === Alias
isstateful(node::PlanNode) = Int(node.kind) & IS_STATEFUL != 0
SyntaxInterface.istree(node::PlanNode) = Int(node.kind) & IS_TREE != 0
AbstractTrees.children(node::PlanNode) = node.children
SyntaxInterface.arguments(node::PlanNode) = node.children
SyntaxInterface.operation(node::PlanNode) = node.kind

function SyntaxInterface.similarterm(x::PlanNode, op::PlanNodeKind, args)
    @assert Int(op) & IS_TREE != 0
    PlanNode(op, args, nothing, deepcopy(x.stats), x.node_id)
end

logic_leaf(arg) = Value(arg)
logic_leaf(arg::Type) = Value(arg)
logic_leaf(arg::Function) = Value(arg)
logic_leaf(arg::Union{Symbol, String}) = Index(arg)
logic_leaf(arg::PlanNode) = arg

Base.convert(::Type{PlanNode}, x) = logic_leaf(x)
Base.convert(::Type{PlanNode}, x::PlanNode) = x

#overload RewriteTools pattern constructor so we don't need
#to wrap leaf nodes.
galley_pattern(arg) = logic_leaf(arg)
galley_pattern(arg::RewriteTools.Slot) = arg
galley_pattern(arg::RewriteTools.Segment) = arg
galley_pattern(arg::RewriteTools.Term) = arg
function RewriteTools.term(f::PlanNodeKind, args...; type = nothing)
    RewriteTools.Term(f, [galley_pattern.(args)...])
end

function Base.getproperty(node::PlanNode, sym::Symbol)
    if sym === :kind || sym === :val || sym === :children || sym == :stats || sym == :node_id
        return Base.getfield(node, sym)
    elseif node.kind === Index && sym === :name node.val
    elseif node.kind === Alias && sym === :name node.val
    elseif node.kind === Input && sym === :tns node.children[1]
    elseif node.kind === Input && sym === :idxs begin length(node.children) > 1 ? node.children[2:end] : [] end
    elseif node.kind === MapJoin && sym === :op node.children[1]
    elseif node.kind === MapJoin && sym === :args @view node.children[2:end]
    elseif node.kind === Aggregate && sym === :op node.children[1]
    elseif node.kind === Aggregate && sym === :idxs begin length(node.children) > 2 ? node.children[2:end-1] : [] end
    elseif node.kind === Aggregate && sym === :arg node.children[end]
    elseif node.kind === Materialize && sym === :formats begin length(node.children) > 1 ? node.children[1:Int((length(node.children)-1)/2)] : [] end
    elseif node.kind === Materialize && sym === :idx_order begin length(node.children) > 1 ? node.children[Int((length(node.children)-1)/2)+1:length(node.children)-1] : [] end
    elseif node.kind === Materialize && sym === :expr node.children[end]
    elseif node.kind === Query && sym === :name node.children[1]
    elseif node.kind === Query && sym === :expr node.children[2]
    elseif node.kind === Query && sym === :loop_order begin length(node.children) > 2 ? node.children[3:end] : [] end
    elseif node.kind === Outputs && sym === :names node.children
    elseif node.kind === Plan && sym === :queries @view node.children[1:end-1]
    elseif node.kind === Plan && sym === :outputs @view node.children[end]
    else
        error("type PlanNode($(node.kind), ...) has no property $sym")
    end
end

function Base.setproperty!(node::PlanNode, sym::Symbol, v)
    if sym === :kind || sym === :val || sym === :children || sym == :stats || sym == :node_id
        return Base.setfield!(node, sym, v)
    elseif node.kind === Index && sym === :name node.val = v
    elseif node.kind === Alias && sym === :name node.val = v
    elseif node.kind === Input && sym === :tns node.children[1] = v
    elseif node.kind === Input && sym === :idxs begin node.children = [node.children[1], v...] end
    elseif node.kind === MapJoin && sym === :op node.children[1] = v
    elseif node.kind === MapJoin && sym === :args begin node.children = [node.children[1], v...] end
    elseif node.kind === Aggregate && sym === :op node.children[1]
    elseif node.kind === Aggregate && sym === :idxs begin node.children = [node.children[1], v..., node.children[end]] end
    elseif node.kind === Aggregate && sym === :arg node.children[end] = v
    elseif node.kind === Materialize && sym === :formats begin node.children = [v..., node.idx_order..., node.expr] end
    elseif node.kind === Materialize && sym === :idx_order begin node.children = [node.formats..., v..., node.expr] end
    elseif node.kind === Materialize && sym === :expr node.children[end] = v
    elseif node.kind === Query && sym === :name node.children[1] = v
    elseif node.kind === Query && sym === :expr node.children[2] = v
    elseif node.kind === Query && sym === :loop_order begin node.children = [node.name, node.expr, v...] end
    elseif node.kind === Outputs && sym === :names node.children = v
    elseif node.kind === Plan && sym === :queries begin node.children = [v..., node.outputs] end
    elseif node.kind === Plan && sym === :outputs node.children[end] = v
    else
        error("type PlanNode($(node.kind), ...) has no property $sym")
    end
end

function relabel_input(input::PlanNode, indices...)
    if input.kind != Input
        throw(ErrorException("Can't relabel a node other than input!"))
    end
    relabeled_input = Input(input.tns, indices...)
    relabeled_input.stats = reindex_stats(input.stats, collect(indices))
    return relabeled_input
end

function Base.:(==)(a::PlanNode, b::PlanNode)
    if a.kind === Value
        if a.val isa Tensor
            return typeof(a.val) == typeof(b.val) # We don't test for tensor equality.
        else
            return typeof(a.val) == typeof(b.val) && a.val == b.val
        end
    elseif a.kind === Alias
        return b.kind === Alias && a.name == b.name
    elseif a.kind === Index
        return b.kind === Index && a.name == b.name
    elseif a.kind == Aggregate
        return b.kind === Aggregate && a.arg == b.arg && Set(a.idxs) == Set(b.idxs)
    elseif istree(a)
        return a.kind === b.kind && a.children == b.children
    else
        error("unimplemented")
    end
end

function Base.hash(a::PlanNode, h::UInt)
    if a.kind === Value || a.kind === Alias || a.kind === Index
        return hash(a.kind, hash(a.val, h))
    elseif istree(a)
        return hash(a.kind, hash(a.children, h))
    else
        error("unimplemented")
    end
end

function planToString(n::PlanNode, depth::Int64)
    output = ""
    if n.kind == Alias
        output *= "Alias($(n.name)"
        if isnothing(n.stats)
            output *= ")"
            return output
        end
        idxs = get_index_order(n.stats)
        protocols = get_index_protocols(n.stats)
        prefix = ","
        if !isnothing(get_index_protocols(n.stats))
            for i in eachindex(idxs)
                output *= "$prefix$(idxs[i])::$(protocols[i])"
            end
        elseif !isnothing(get_index_set(n.stats))
            for idx in get_index_set(n.stats)
                output *= prefix * string(idx)
            end
        end
        output *= ")"
        return output
    elseif n.kind == Index
        output *= "$(n.name)"
        return output
    elseif n.kind == Value
        if n.val isa Tensor
            output *= "Value(FIBER)"
        else
            output *= "Value($(n.val))"
        end
        return output
    elseif n.kind == Input
        output *= "Input("
        prefix = ""
        idxs = n.idxs
        if !isnothing(n.stats) && !isnothing(get_index_protocols(n.stats))
            protocols = get_index_protocols(n.stats)
            for i in eachindex(idxs)
                output *= "$prefix$(idxs[i])::$(protocols[i])"
                prefix =","
            end
        else
            for arg in children(n)
                output *= prefix * planToString(arg, depth + 1)
                prefix =","
            end
        end
        output *= ")"
        return output
    end

    if depth > 0
        output = "\n"
    end
    left_space = ""
    for _ in 1:depth
        left_space *= "   "
    end
    output *= left_space
    if n.kind == Aggregate
        output *= "Aggregate("
    elseif n.kind == MapJoin
        output *= "MapJoin("
    elseif n.kind == Materialize
        output *= "Materialize("
    elseif n.kind == Query
        output *= "Query("
    elseif n.kind == Outputs
        output *= "Outputs("
    elseif n.kind == Plan
        output *= "Plan("
    end

    prefix = ""
    for arg in children(n)
        output *= prefix * planToString(arg, depth + 1)
        prefix =","
    end
    output *= ")"
end

function Base.show(io::IO, input::PlanNode)
    print(io, planToString(input, 0))
end
