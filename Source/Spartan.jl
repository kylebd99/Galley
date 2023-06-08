# This file defines a prototype front-end which allows users to define tensor expressions and get their results.
using Metatheory
using Metatheory.EGraphs
using PrettyPrinting
include("LogicalOptimizer.jl")
include("ExecutionEngine.jl")

function get_index_order(expr)
    if expr isa Vector{String}
        return expr
    elseif expr isa LogicalPlanNode
        return sort(union([get_index_order(child) for child in expr.args]...))
    else
        return []
    end
end

function spartan(expr; optimize=true, verbose=2, global_index_order=[])
    if global_index_order == [] 
        global_index_order = get_index_order(expr)
    end
    expr = insertInputReorders(expr, global_index_order)
    expr = insertGlobalOrders(expr, global_index_order)
    expr = removeUnecessaryReorders(expr, global_index_order)

    if optimize
        g = EGraph(expr)
        settermtype!(g, LogicalPlanNode)
        analyze!(g, :TensorStatsAnalysis)
        params = SaturationParams(timeout=100, eclasslimit=4000)
        saturation_report = saturate!(g, basic_rewrites, params);
        if verbose >=2
            println(saturation_report)
        end
        expr = extract!(g, simple_cardinality_cost_function)
    end
    expr = mergeAggregates(expr)

    if verbose >= 1
        optimize && print("Optimized Expression: ")
        !optimize && print("Expression: ")
        println(expr)
    end
    g = EGraph(expr)
    settermtype!(g, LogicalPlanNode)
    analyze!(g, :TensorStatsAnalysis)

    expr_tree = e_graph_to_expr_tree(g, global_index_order)
    tensor_kernel = expr_to_kernel(expr_tree, global_index_order, verbose = verbose)

    result = @timed execute_tensor_kernel(tensor_kernel, verbose = verbose)

    verbose >= 1 && println("Time to Execute: ", result.time)

    return result.value
end