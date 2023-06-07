# This file defines a prototype front-end which allows users to define tensor expressions and get their results.
using Metatheory
using Metatheory.EGraphs
using PrettyPrinting
include("LogicalOptimizer.jl")
include("ExecutionEngine.jl")

function get_index_order(expr)
    g = EGraph(expr)
    analyze!(g, :TensorStatsAnalysis)
    all_indices = Set()
    for class in values(g.classes)
        for idx in getdata(class, :TensorStatsAnalysis).indices
            push!(all_indices, idx)
        end
    end
    return sort(collect(all_indices))
end

function spartan(expr; optimize=true, verbose=2, global_index_order=[])
    if global_index_order == [] 
        global_index_order = get_index_order(expr)
    end
    expr = insertInputReorders(expr, global_index_order)
    expr = insertGlobalOrders(expr, global_index_order)
    if optimize
        g = EGraph(expr)
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
        optimize && println("Optimized Expression: ")
        !optimize && println("Expression: ")
        pprintln(expr)
    end
    g = EGraph(expr)
    analyze!(g, :TensorStatsAnalysis)

    expr_tree = label_expr_parents!(nothing, e_graph_to_expr_tree(g, global_index_order))
    tensor_kernel = expr_to_kernel(expr_tree, global_index_order, verbose = verbose)

    result = @timed execute_tensor_kernel(tensor_kernel, verbose = verbose)

    verbose >= 1 && println("Time to Execute: ", result.time)

    return result.value
end