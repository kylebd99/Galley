mutable struct GalleyOptimizer
    estimator
    verbose
end

GalleyOptimizer(; verbose = false, estimator=DCStats) = GalleyOptimizer(estimator, verbose)

function (ctx::GalleyOptimizer)(prgm)
    finch_mode = ctx.verbose ? :safe : :fast
    verbosity = ctx.verbose ? 3 : 0
    produce_node = prgm.bodies[end]
    output_vars = [Alias(a.name) for a in produce_node.args]
    galley_prgm = Plan(finch_hl_to_galley(normalize_hl(prgm))...)
    tns_inits, instance_prgm = galley(galley_prgm, ST=ctx.estimator, output_aliases=output_vars, verbose=verbosity, output_program_instance=true)
    julia_prgm = :()
    if operation(instance_prgm) == Finch.block
        for body in instance_prgm.bodies
            julia_prgm = :($julia_prgm; @finch mode=$(QuoteNode(finch_mode)) begin $(finch_unparse_program(nothing, body)) end)
        end
    else
        julia_prgm = :(@finch mode=$(QuoteNode(finch_mode)) begin $(finch_unparse_program(nothing, instance_prgm)) end)
    end
    for init in tns_inits
        julia_prgm = :($init; $julia_prgm)
    end
    julia_prgm = :($julia_prgm; return Tuple([$([v.name for v in output_vars]...)]))
    julia_prgm
end

function Finch.set_options(ctx::GalleyOptimizer; estimator=DCStats)
    ctx.estimator=estimator
    return ctx
end

galley_scheduler(; verbose = false, estimator=DCStats) = Finch.LogicExecutor(GalleyOptimizer(verbose=verbose, estimator=estimator); verbose=verbose)

# Roadmap:
#   - Register Galley as a julia package (juliaregistrator, tagbot) @Kyle
#   - Merge the Finch PR @Kyle
#       - Add finch_unparse_program tests @Kyle
#       - Add downstream Galley tests to CI.yml (Refer to Galley/main) @Kyle
#   - Add Documentation in Finch (need simple shortcuts Compute(...,ctx=Galley())) @Willow
#   - Add PythonPR to add compute() param for optimizers @Kyle
#       - Discussion of adding Galley to python deps
#       - Minimal param interface to enable Galley 
#       - Get perftest file 
#   - Add precompilation for Galley @Kyle
