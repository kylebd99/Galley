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
export PlanNode, Value, Index, Alias, Input, MapJoin, Aggregate, Materialize, Query, Outputs, Plan
export Scalar, OutTensor, RenameIndices, declare_binary_operator, ∑, ∏
export Factor, FAQInstance, Bag, HyperTreeDecomposition, decomposition_to_logical_plan
export DCStats, NaiveStats, TensorDef, DC, insert_statistics
export naive, hypertree_width, greedy, ordering
export expr_to_kernel, execute_tensor_kernel
export load_to_duckdb, DuckDBTensor

IndexExpr = Symbol
TensorId = String
# This defines the list of access protocols allowed by the Finch API
@enum AccessProtocol t_walk = 1 t_lead = 2 t_follow = 3 t_gallop = 4 t_default = 5
# A subset of the allowed level formats provided by the Finch API
@enum LevelFormat t_sparse_list = 1 t_dense = 2 t_hash = 3 t_undef = 4
# The set of optimizers implemented by Galley
@enum FAQ_OPTIMIZERS greedy

include("finch-algebra_ext.jl")
include("utility-funcs.jl")
include("PlanAST/PlanAST.jl")
include("TensorStats/TensorStats.jl")
include("FAQOptimizer/FAQOptimizer.jl")
include("PhysicalOptimizer/PhysicalOptimizer.jl")
include("ExecutionEngine/ExecutionEngine.jl")



function galley(input_query::PlanNode;
                    faq_optimizer::FAQ_OPTIMIZERS=greedy,
                    ST=DCStats,
                    dbconn::Union{DuckDB.DB, Nothing}=nothing,
                    verbose=0)
#    verbose >= 3 && println("Input FAQ : ", faq_problem)
    opt_start = time()
    faq_opt_start = time()
    aq = AnnotatedQuery(input_query, ST)
    logical_plan = greedy_aq_to_plan(aq)
    faq_opt_end = time()
    verbose >= 1 && println("FAQ Opt Time: $(faq_opt_end-faq_opt_start)")
    if !isnothing(dbconn)
        return
#=      opt_end = time()
        result = duckdb_htd_to_output(dbconn, htd)
        verbose >= 1 && println("Plan: ", expr)
        verbose >= 1 && println("Time to Optimize: ", (opt_end-opt_start))
        verbose >= 1 && println("Time to Insert: ", result.insert_time)
        verbose >= 1 && println("Time to Execute: ", result.execute_time)
        return (value=result.value,
                    opt_time=(opt_end-opt_start + result.opt_time),
                    execute_time=result.execute_time)=#
    end
    println(logical_plan)
    alias_stats = Dict{PlanNode, TensorStats}()
    physical_queries = []
    for query in logical_plan.queries
        translated_queries = logical_query_to_physical_queries(alias_stats, query)
        append!(physical_queries, translated_queries)
    end
    opt_end = time()
    verbose >= 1 && println("Physical Opt Time: $(opt_end - faq_opt_end)")

    alias_stats = Dict()
    alias_result = Dict()
    for query in physical_queries
#        println(query)
#        println(keys(alias_result))
        validate_physical_query(query, alias_stats)
        execute_query(alias_result, query, verbose)
    end
    exec_end = time()
    verbose >= 1 && println("Time to Optimize: ", (opt_end-opt_start))
    verbose >= 1 && println("Time to Execute: ", (exec_end - opt_end))
    return (value=alias_result[physical_queries[end].name], opt_time=(opt_end-opt_start), execute_time= (exec_end - opt_end))
end

end
