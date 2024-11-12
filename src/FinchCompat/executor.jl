using Finch: set_options
using Finch: compute_parse, isimmediate
using Finch.FinchNotation: finch_unparse_program

mutable struct GalleyOptimizer
    verbose
end

GalleyOptimizer(; verbose = false) = GalleyOptimizer(verbose)

function (ctx::GalleyOptimizer)(prgm)
    produce_node = prgm.bodies[end]
    output_vars = [Alias(a.name) for a in produce_node.args]
    galley_prgm = Plan(finch_hl_to_galley(normalize_hl(prgm))...)
    tns_inits, instance_prgm = galley(galley_prgm, output_aliases=output_vars, verbose=0, output_program_instance=true)
    julia_prgm = :(@finch begin $(finch_unparse_program(ctx, instance_prgm)) end)
    for init in tns_inits
        julia_prgm = :($init; $julia_prgm)
    end
    julia_prgm = :($julia_prgm; return Tuple([$([v.name for v in output_vars]...)]))
    julia_prgm
end

function Finch.set_options(ctx::GalleyOptimizer; verbose=false)
    ctx.verbose=verbose
    return ctx
end
