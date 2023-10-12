
# This file defines a prototype front-end which allows users to define tensor expressions and get their results.
module Galley
    export galley, InputTensor, IndexExpr, TensorStats, OutTensor, ∑, ∏, Aggregate, MapJoin, Scalar, Agg
    export uniform_fiber, declare_binary_operator

    using AutoHashEquals
    using Combinatorics
    using DataStructures
    using Finch
    using Finch: @finch_program_instance, SparseHashLevel
    using Metatheory
    using Metatheory.EGraphs
    using PrettyPrinting
    using TermInterface


    include("utility-funcs.jl")
    include("LogicalOptimizer/LogicalOptimizer.jl")
    include("PhysicalOptimizer/PhysicalOptimizer.jl")
    include("ExecutionEngine/ExecutionEngine.jl")


    function get_index_order(expr, perm_choice=-1)
        if expr isa Vector{IndexExpr}
            return expr
        elseif expr isa LogicalPlanNode && perm_choice > 0
            return nthperm(sort(union([get_index_order(child) for child in expr.args]...)), perm_choice)
        elseif expr isa LogicalPlanNode
            return sort(union([get_index_order(child) for child in expr.args]...))

        else
            return []
        end
    end

    function fill_in_stats(expr, global_index_order)
        g = EGraph(expr)
        settermtype!(g, LogicalPlanNode)
        analyze!(g, :TensorStatsAnalysis)
        expr = e_graph_to_expr_tree(g, global_index_order)
        return expr
    end


    function galley(expr; optimize=true, verbose=2, global_index_order=1)
        verbose >= 3 && println("Before Rename Pass: ", expr)
        dummy_index_order = get_index_order(expr)
        expr = insert_global_orders(expr, dummy_index_order)
        expr = fill_in_stats(expr, dummy_index_order)
        expr = recursive_rename(expr, Dict(), 0, 0, [0], true, true)

        verbose >= 3 && println("After Rename Pass: ", expr)
        global_index_order = get_index_order(expr, global_index_order)
        expr = insert_input_reorders(expr, global_index_order)
        expr = insert_global_orders(expr, global_index_order)
        expr = remove_uneccessary_reorders(expr, global_index_order)

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
        expr = merge_aggregates(expr)

        if verbose >= 1
            optimize && print("Optimized Expression: ")
            !optimize && print("Expression: ")
            println(expr)
        end

        expr = fill_in_stats(expr, global_index_order)
        tensor_kernel = expr_to_kernel(expr, global_index_order, verbose = verbose)
        result = @timed execute_tensor_kernel(tensor_kernel, verbose = verbose)

        verbose >= 1 && println("Time to Execute: ", result.time)

        return result.value
    end
end
