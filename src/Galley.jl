# This file defines a prototype front-end which allows users to define tensor expressions and get their results.
module Galley

using AutoHashEquals
using Combinatorics
using DataStructures
using Random
using Profile
using IterTools: subsets
using RewriteTools
using RewriteTools.Rewriters
using SyntaxInterface
using AbstractTrees
using Finch
using Finch: @finch_program_instance, Element, SparseListLevel, Dense, SparseHashLevel, SparseCOO, fsparse_impl
using Finch.FinchNotation: index_instance, variable_instance, tag_instance, literal_instance,
                        access_instance,  assign_instance, loop_instance, declare_instance,
                        block_instance, define_instance, call_instance, freeze_instance,
                        thaw_instance,
                        Updater, Reader, Dimensionless
using DuckDB
using PrettyPrinting

export galley
export PlanNode, Value, Index, Alias, Input, MapJoin, Aggregate, Materialize, Query, Outputs, Plan, IndexExpr
export Scalar, OutTensor, RenameIndices, declare_binary_operator, ∑, ∏
export Factor, FAQInstance, Bag, HyperTreeDecomposition, decomposition_to_logical_plan
export DCStats, NaiveStats, TensorDef, DC, insert_statistics
export naive, hypertree_width, greedy, ordering
export expr_to_kernel, execute_tensor_kernel
export load_to_duckdb, DuckDBTensor, fill_table

IndexExpr = Symbol
TensorId = String
# This defines the list of access protocols allowed by the Finch API
@enum AccessProtocol t_walk = 1 t_lead = 2 t_follow = 3 t_gallop = 4 t_default = 5
# A subset of the allowed level formats provided by the Finch API
@enum LevelFormat t_sparse_list = 1 t_dense = 2 t_hash = 3 t_bytemap = 4 t_undef = 5
# The set of optimizers implemented by Galley
@enum FAQ_OPTIMIZERS greedy naive

include("finch-algebra_ext.jl")
include("utility-funcs.jl")
include("PlanAST/PlanAST.jl")
include("TensorStats/TensorStats.jl")
include("FAQOptimizer/FAQOptimizer.jl")
include("PhysicalOptimizer/PhysicalOptimizer.jl")
include("ExecutionEngine/ExecutionEngine.jl")


# InputQuery: Query(name, Materialize(formats..., idxs..., agg_map_expr))
# Aggregate(op, idxs.., expr)
# MapJoin(op, exprs...)
# TODO:
#   - Convert a Finch HL query to a galley query
#   - On Finch Side:
#           - One query at a time to galley
#           - Isolate reformat_stats
#           - Fuse mapjoins & permutations
function galley(input_query::PlanNode;
                    faq_optimizer::FAQ_OPTIMIZERS=greedy,
                    ST=DCStats,
                    dbconn::Union{DuckDB.DB, Nothing}=nothing,
                    verbose=0)
    input_query = plan_copy(input_query)
    verbose >= 2 && println("Input Query : ", input_query)
    opt_start = time()
    faq_opt_start = time()
    output_order = input_query.expr.idx_order
    logical_plan = high_level_optimize(faq_optimizer, input_query, ST)
    faq_opt_end = time()
    verbose >= 1 && println("FAQ Opt Time: $(faq_opt_end-faq_opt_start)")
    split_start = time()
    if faq_optimizer != naive
        logical_plan = split_queries(ST, logical_plan)
    end
    split_end = time()
    verbose >= 1 && println("Split Opt Time: $(split_end-split_start)")

    # TODO: Add the step which splits up overly complex kernels back in
    if verbose >= 1
        println("--------------- Logical Plan ---------------")
        println(logical_plan)
        println("--------------------------------------------")
    end
    if !isnothing(dbconn)
        verbose
        opt_end = time()
        output_order = input_query.expr.idx_order
        duckdb_opt_time = (opt_end-opt_start)
        duckdb_exec_time = 0
        duckdb_insert_time = 0
        for query in logical_plan.queries
            verbose >= 1 && println("-------------- Computing Alias $(query.name) -------------")
            query_timings = duckdb_execute_query(dbconn, query, verbose)
            verbose >= 1 && println("$query_timings")
            duckdb_opt_time += query_timings.opt_time
            duckdb_exec_time += query_timings.execute_time
            duckdb_insert_time += query_timings.insert_time
        end
        result = _duckdb_query_to_tns(dbconn, logical_plan.queries[end], output_order)
        for query in logical_plan.queries
            _duckdb_drop_alias(dbconn, query.name)
        end
        verbose >= 1 && println("Time to Optimize: ",  duckdb_opt_time)
        verbose >= 1 && println("Time to Insert: ", duckdb_insert_time)
        verbose >= 1 && println("Time to Execute: ", duckdb_exec_time)
        return (value=result,
                    opt_time=duckdb_opt_time,
                    insert_time = duckdb_insert_time,
                    execute_time=duckdb_exec_time)
    end
    alias_stats = Dict{PlanNode, TensorStats}()
    physical_queries = []
    for query in logical_plan.queries
        translated_queries = logical_query_to_physical_queries(alias_stats, query)
        append!(physical_queries, translated_queries)
    end

    # Determine the optimal access protocols for every index occurence
    alias_stats = Dict{PlanNode, TensorStats}()
    for query in physical_queries
        insert_node_ids!(query)
        input_stats = get_input_stats(alias_stats, query.expr)
        modify_protocols!(collect(values(input_stats)))
        alias_stats[query.name] = query.expr.stats
    end

    opt_end = time()
    verbose >= 1 && println("Physical Opt Time: $(opt_end - faq_opt_end)")
    if verbose >= 2
        println("--------------- Physical Plan ---------------")
        for query in physical_queries
            println(query)
        end
        println("--------------------------------------------")
    end
    alias_result = Dict()
    for query in physical_queries
        verbose > 3 && validate_physical_query(query)
        execute_query(alias_result, query, verbose)
    end
    exec_end = time()
    verbose >= 1 && println("Time to Optimize: ", (opt_end-opt_start))
    verbose >= 1 && println("Time to Execute: ", (exec_end - opt_end))
    return (value=alias_result[physical_queries[end].name], opt_time=(opt_end-opt_start), execute_time= (exec_end - opt_end))
end

end
