using Finch: set_options
using Finch: compute_parse, isimmediate


"""
    defer_tables(root::LogicNode)

Replace immediate tensors with deferred expressions assuming the original program structure
is given as input to the program.
"""
function defer_tables(ex, node::LogicNode)
    if @capture node table(~tns::isimmediate, ~idxs...)
        table(deferred(:($ex.tns.val), tns.val), map(enumerate(node.idxs)) do (i, idx)
            defer_tables(:($ex.idxs[$i]), idx)
        end)
    elseif istree(node)
        similarterm(node, operation(node), map(enumerate(node.children)) do (i, child)
            defer_tables(:($ex.children[$i]), child)
        end)
    else
        node
    end
end

"""
    cache_deferred(ctx, root::LogicNode, seen)

Replace deferred expressions with simpler expressions, and cache their evaluation in the preamble.
"""
function cache_deferred!(ctx, root::LogicNode)
    seen::Dict{Any, LogicNode} = Dict{Any, LogicNode}()
    return Rewrite(Postwalk(node -> if isdeferred(node)
        get!(seen, node.val) do
            var = freshen(ctx, :V)
            push_preamble!(ctx, :($var = $(node.ex)::$(typeof(node.type))))
            deferred(var, node.type)
        end
    end))(root)
end

function galley_executor_code(ctx, prgm)
    ctx_2 = Finch.JuliaContext()
    freshen(ctx_2, :prgm)
    code = contain(ctx_2) do ctx_3
        prgm = defer_tables(:prgm, prgm)
        prgm = cache_deferred!(ctx_3, prgm)
        ctx(prgm)
    end
    code = pretty(code)
    fname = gensym(:compute)
    return :(function $fname(prgm)
            $code
        end)
end

mutable struct GalleyExecutor
    verbose
end

GalleyExecutor(; verbose = false) = GalleyExecutor(verbose)

codes = Dict()
function (ctx::GalleyExecutor)(prgm)
    initial_prgm = deepcopy(prgm)
    produce_node = prgm.bodies[end]
    output_vars = [Alias(a.name) for a in produce_node.args]
    ctx_2 = Finch.JuliaContext()
    code = Finch.contain(ctx_2) do ctx_3
        prgm = Finch.defer_tables(:prgm, prgm)
        prgm = Finch.cache_deferred!(ctx_3, prgm)
        galley_prgm = Plan(finch_hl_to_galley(normalize_hl(prgm))...)
        tns_inits, instance_prgm = galley(galley_prgm, output_aliases=output_vars, verbose=0, output_program_instance=true)
        julia_prgm = :(@finch begin $(finch_unparse_program(ctx, instance_prgm)) end)
        for init in tns_inits
            julia_prgm = :($init; $julia_prgm)
        end
        julia_prgm = :($julia_prgm; return Tuple([$([v.name for v in output_vars]...)]))
        julia_prgm
    end
    code = Finch.pretty(code)
    fname = gensym(:compute)
    #= println(:(function $fname(prgm)
                  $code
              end)) =#
    eval(:(function $fname(prgm) $code end))
    #= println(eval(:($fname($initial_prgm)))) =#
    return eval(:($fname($initial_prgm)))
    if length(galley_prgm.queries) == 1
        return galley(galley_prgm, output_aliases=output_vars, verbose=0).value
    else
        return tuple(galley(galley_prgm, output_aliases=output_vars, verbose=0).value...)
    end
end

#=
begin
    (f, code) = get!(codes, get_structure(prgm)) do
        thunk = logic_executor_code(ctx.ctx, prgm)
        (eval(thunk), thunk)
    end
    if ctx.verbose
        println("Executing:")
        display(code)
    end
    return Base.invokelatest(f, prgm)
end =#


function Finch.set_options(ctx::GalleyExecutor; verbose=false)
    ctx.verbose=verbose
    return ctx
end

finch_unparse_program(ctx, node) = finch_unparse_program(ctx, Finch.finch_leaf(node))
function finch_unparse_program(ctx, node::Union{Finch.FinchNode, Finch.FinchNotation.FinchNodeInstance})
    if operation(node) === Finch.value
        node.val
    elseif operation(node) === Finch.literal
        node.val
    elseif operation(node) === Finch.index
        node.name
    elseif operation(node) === Finch.variable
        node.name
    elseif operation(node) === Finch.cached
        finch_unparse_program(ctx, node.arg)
    elseif operation(node) === Finch.tag
        @assert operation(node.var) === Finch.variable
        node.var.name
    elseif operation(node) === Finch.virtual
        if node.val == Finch.dimless
            :_
        else
            ctx(node)
        end
    elseif operation(node) === Finch.access
        tns = finch_unparse_program(ctx, node.tns)
        idxs = map(x -> finch_unparse_program(ctx, x), node.idxs)
        :($tns[$(idxs...)])
    elseif operation(node) === Finch.call
        op = finch_unparse_program(ctx, node.op)
        args = map(x -> finch_unparse_program(ctx, x), node.args)
        :($op($(args...)))
    elseif operation(node) === Finch.loop
        idx = finch_unparse_program(ctx, node.idx)
        ext = finch_unparse_program(ctx, node.ext)
        body = finch_unparse_program(ctx, node.body)
        :(for $idx = $ext; $body end)
    elseif operation(node) === Finch.define
        lhs = finch_unparse_program(ctx, node.lhs)
        rhs = finch_unparse_program(ctx, node.rhs)
        body = finch_unparse_program(ctx, node.body)
        :(let $lhs = $rhs; $body end)
    elseif operation(node) === Finch.sieve
        cond = finch_unparse_program(ctx, node.cond)
        body = finch_unparse_program(ctx, node.body)
        :(if $cond; $body end)
    elseif operation(node) === Finch.assign
        lhs = finch_unparse_program(ctx, node.lhs)
        op = finch_unparse_program(ctx, node.op)
        rhs = finch_unparse_program(ctx, node.rhs)
        if haskey(Finch.incs, op)
            Expr(incs[op], lhs, rhs)
        else
            :($lhs <<$op>>= $rhs)
        end
    elseif operation(node) === Finch.declare
        tns = finch_unparse_program(ctx, node.tns)
        init = finch_unparse_program(ctx, node.init)
        :($tns .= $init)
    elseif operation(node) === Finch.freeze
        tns = finch_unparse_program(ctx, node.tns)
        :(@freeze($tns))
    elseif operation(node) === Finch.thaw
        tns = finch_unparse_program(ctx, node.tns)
        :(@thaw($tns))
    elseif operation(node) === Finch.yieldbind
        args = map(x -> finch_unparse_program(ctx, x), node.args)
        :(return($(args...)))
    elseif operation(node) === Finch.block
        bodies = map(x -> finch_unparse_program(ctx, x), node.bodies)
        Expr(:block, bodies...)
    else
        error("unimplemented")
    end
end
