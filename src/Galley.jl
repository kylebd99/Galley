
# This file defines a prototype front-end which allows users to define tensor expressions and get their results.
module Galley
    export galley, InputTensor, IndexExpr, TensorStats, NaiveStats, OutTensor, ∑, ∏, Aggregate, MapJoin, Scalar, Agg
    export uniform_fiber, declare_binary_operator, Factor, FAQInstance
    export FAQ_OPTIMIZERS, naive, hypertree_width, greedy

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
    include("TensorStats/TensorStats.jl")
    include("FAQOptimizer/FAQOptimizer.jl")
    include("PhysicalOptimizer/PhysicalOptimizer.jl")
    include("ExecutionEngine/ExecutionEngine.jl")


    function galley(expr::LogicalPlanNode; optimize=true, verbose=0, global_index_order=1)
        verbose >= 3 && println("Before Rename Pass: ", expr)
        dummy_index_order = get_index_order(expr)
        expr = insert_global_orders(expr, dummy_index_order)
        expr = fill_in_stats(expr, dummy_index_order)
        expr = recursive_rename(expr, Dict(), 0, 0, [0], true, true)

        verbose >= 3 && println("After Rename Pass: ", expr)
        global_index_order = get_index_order(expr, global_index_order)
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



    function galley(faq_problem::FAQInstance; faq_optimizer::FAQ_OPTIMIZERS=naive, verbose=0)
        verbose >= 3 && println("Input FAQ : ", faq_problem)

        htd = faq_to_htd(faq_problem; faq_optimizer=faq_optimizer)
        expr = decomposition_to_logical_plan(htd)
        expr = merge_aggregates(expr)
        _recursive_insert_stats!(expr)
        verbose >= 1 && println("Plan: ", expr)
        output_index_order = htd.output_index_order
        if isnothing(htd.output_index_order)
            output_index_order = collect(htd.output_indices)
        end

        tensor_kernel = expr_to_kernel(expr, output_index_order, verbose = verbose)
        result = @timed execute_tensor_kernel(tensor_kernel, verbose = verbose)
        verbose >= 1 && println("Time to Execute: ", result.time)
        return result.value
    end

end
