using Finch: set_options
using Finch: compute_parse

mutable struct GalleyExecutor

end

function Finch.set_options(ctx::GalleyExecutor; kwargs...)
    return ctx
end

function Finch.compute_parse(ctx::GalleyExecutor, args::Tuple)
    args = collect(args)
    vars = map(arg -> alias(gensym(:A)), args)
    bodies = map((arg, var) -> query(var, arg.data), args, vars)
    galley_prgm = finch_hl_to_galley(normalize_hl(plan(bodies...)))
    if length(galley_prgm) == 1
        return galley(galley_prgm, output_aliases=vars, verbose=0).value
    else
        return tuple(galley(galley_prgm, output_aliases=vars, verbose=0).value...)
    end
end
