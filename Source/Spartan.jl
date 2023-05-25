# This file defines a prototype front-end which allows users to define tensor expressions and get their results.
using Metatheory
using Metatheory.EGraphs
using PrettyPrinting
include("LogicalOptimizer.jl")
include("ExecutionEngine.jl")

function spartan(expr, input_tensors; optimize=true, verbose=2)
    if optimize
        g = EGraph(expr)
        analyze!(g, :TensorStatsAnalysis)
        params = SaturationParams(timeout=100, eclasslimit=4000)
        saturation_report = saturate!(g, basic_rewrites, params);
        if verbose >=1
            println(saturation_report)
        end
        expr = extract!(g, simple_cardinality_cost_function)
    end
    if verbose >= 1
        optimize && println("Optimized Expression: ")
        !optimize && println("Expression: ")
        pprintln(expr)
    end
    g = EGraph(expr)

    analyze!(g, :TensorStatsAnalysis)
    expr_tree = label_expr_parents!(nothing, e_graph_to_expr_tree(g))
    tensor_kernel = expr_to_kernel(expr_tree, input_tensors, verbose = verbose)

    result = @timed execute_tensor_kernel(tensor_kernel, verbose = verbose)

    verbose >= 1 && println("Time to Execute: ", result.time)

    return result.value
end