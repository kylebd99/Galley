# This file defines a prototype front-end which allows users to define tensor expressions and get their results.
module Galley

using AutoHashEquals
using Combinatorics
using DataStructures
using Finch
using Finch: @finch_program_instance, Element, SparseListLevel, Dense, SparseHashLevel, SparseCOO, fsparse_impl
using Random
using Profile
using Finch.FinchNotation: index_instance, variable_instance, tag_instance, literal_instance,
                        access_instance,  assign_instance, loop_instance, declare_instance,
                        block_instance, define_instance, call_instance, freeze_instance,
                        thaw_instance,
                        Updater, Reader, Dimensionless
using Metatheory
using Metatheory.EGraphs
using PrettyPrinting
using TermInterface

export galley
export LogicalPlanNode, IndexExpr, Aggregate, Agg, MapJoin, Reorder, InputTensor
export Scalar, OutTensor, RenameIndices, declare_binary_operator, ∑, ∏
export Factor, FAQInstance, Bag, HyperTreeDecomposition, decomposition_to_logical_plan
export DCStats, NaiveStats, _recursive_insert_stats!, TensorDef, DC
export naive, hypertree_width, greedy, ordering
export expr_to_kernel, execute_tensor_kernel
export load_to_duckdb, DuckDBTensor

include("finch-algebra_ext.jl")
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

function galley(faq_problem::FAQInstance;
                    faq_optimizer::FAQ_OPTIMIZERS=naive,
                    dbconn::Union{DuckDB.DB, Nothing}=nothing,
                    verbose=0)
    verbose >= 3 && println("Input FAQ : ", faq_problem)
    opt_start = time()

    htd = faq_to_htd(faq_problem; faq_optimizer=faq_optimizer)

    if !isnothing(dbconn)
        opt_end = time()
        result = duckdb_htd_to_output(dbconn, htd)
        verbose >= 1 && println("Plan: ", expr)
        verbose >= 1 && println("Time to Optimize: ", (opt_end-opt_start))
        verbose >= 1 && println("Time to Insert: ", result.insert_time)
        verbose >= 1 && println("Time to Execute: ", result.execute_time)
        return (value=result.value,
                    opt_time=(opt_end-opt_start + result.opt_time),
                    execute_time=result.execute_time)
    end

    expr = decomposition_to_logical_plan(htd)
    expr = merge_aggregates(expr)
    _recursive_insert_stats!(expr)
    verbose >= 1 && println("Plan: ", expr)
    output_index_order = htd.output_index_order
    if isnothing(htd.output_index_order)
        output_index_order = collect(htd.output_indices)
    end
    tensor_kernel = expr_to_kernel(expr, output_index_order, verbose = verbose)
    opt_end = time()

    result = @timed execute_tensor_kernel(tensor_kernel, verbose = verbose)
    verbose >= 1 && println("Time to Optimize: ", (opt_end-opt_start))
    verbose >= 1 && println("Time to Execute: ", result.time)
    return (value=result.value, opt_time=(opt_end-opt_start), execute_time=result.time)
end

end
