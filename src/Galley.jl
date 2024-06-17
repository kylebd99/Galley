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
using Statistics
using Finch
using Finch: Element, SparseListLevel, SparseDict, Dense, SparseCOO, fsparse_impl
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
export naive, hypertree_width, greedy, pruned, exact
export expr_to_kernel, execute_tensor_kernel
export load_to_duckdb, DuckDBTensor, fill_table

IndexExpr = Symbol
TensorId = String
# This defines the list of access protocols allowed by the Finch API
@enum AccessProtocol t_walk = 1 t_lead = 2 t_follow = 3 t_gallop = 4 t_default = 5
# A subset of the allowed level formats provided by the Finch API
@enum LevelFormat t_sparse_list = 1 t_dense = 2 t_hash = 3 t_bytemap = 4 t_undef = 5
# The set of optimizers implemented by Galley
@enum FAQ_OPTIMIZERS greedy naive pruned exact

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
function galley(input_queries::Vector{PlanNode};
                    faq_optimizer::FAQ_OPTIMIZERS=greedy,
                    ST=DCStats,
                    dbconn::Union{DuckDB.DB, Nothing}=nothing,
                    update_cards=true,
                    simple_cse=true,
                    max_kernel_size=5,
                    verbose=0)
    overall_start = time()
    input_queries = map(plan_copy, input_queries)
    if verbose >= 2
        println("Input Queries : ")
        for input_query in input_queries
            println(input_query)
        end
    end
    opt_start = time()
    faq_opt_start = time()
    logical_queries = []
    alias_stats = Dict{PlanNode, TensorStats}()
    alias_hash = Dict{PlanNode, UInt}()
    output_aliases = [input_query.name for input_query in input_queries]
    output_orders = Dict(input_query.name => input_query.expr.idx_order for input_query in input_queries)
    for input_query in input_queries
        check_dnf = !allequal([n.op.val for n in PostOrderDFS(input_query) if n.kind === MapJoin])
        logical_plan, cnf_cost = high_level_optimize(faq_optimizer, input_query, ST, alias_stats, alias_hash, false)
        if check_dnf
            dnf_plan, dnf_cost = high_level_optimize(faq_optimizer, input_query, ST, alias_stats, alias_hash,true)
            logical_plan = dnf_cost < cnf_cost ? dnf_plan : logical_plan
            verbose >= 1 && println("Used DNF: $(dnf_cost < cnf_cost)")
        end
        for query in logical_plan
            alias_hash[query.name] = cannonical_hash(query.expr, alias_hash)
            alias_stats[query.name] = query.expr.stats
        end
        append!(logical_queries, logical_plan)
    end
    faq_opt_time = time() - faq_opt_start
    verbose >= 1 && println("FAQ Opt Time: $faq_opt_time")

    if verbose >= 1
        println("--------------- Logical Plan ---------------")
        for query in logical_queries
            println(query)
        end
        println("--------------------------------------------")
    end
    if !isnothing(dbconn)
        verbose
        opt_end = time()
        output_name = only(output_aliases)
        output_order = output_orders[output_name]
        duckdb_opt_time = (opt_end-opt_start)
        duckdb_exec_time = 0
        duckdb_insert_time = 0
        for query in logical_queries
            verbose >= 1 && println("-------------- Computing Alias $(query.name) -------------")
            query_timings = duckdb_execute_query(dbconn, query, verbose)
            verbose >= 1 && println("$query_timings")
            duckdb_opt_time += query_timings.opt_time
            duckdb_exec_time += query_timings.execute_time
            duckdb_insert_time += query_timings.insert_time
        end
        result = _duckdb_query_to_tns(dbconn, logical_queries[end], output_order)
        for query in logical_queries
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
    opt_end = time()
    total_split_time = 0
    total_phys_opt_time = 0
    total_exec_time = 0
    total_count_time = 0
    plan_hash_result = Dict()
    alias_result = Dict()
    for l_query in logical_queries
        split_start = time()
        split_queries = split_query(l_query, ST, max_kernel_size, alias_stats)
        total_split_time  += time() - split_start
        for s_query in split_queries
            phys_opt_start = time()
            physical_queries = logical_query_to_physical_queries(s_query, ST, alias_stats)
            total_phys_opt_time += time() - phys_opt_start
            for p_query in physical_queries
                phys_opt_start = time()
                modify_protocols!(p_query.expr)
                total_phys_opt_time += time() - phys_opt_start
                alias_stats[p_query.name] = p_query.expr.stats

                verbose > 2 && println("--------------- Computing: $(p_query.name) ---------------")
                verbose > 2 && println(p_query)
                verbose > 4 && validate_physical_query(p_query)
                exec_start = time()
                p_query_hash = cannonical_hash(p_query.expr, alias_hash)
                alias_hash[p_query.name] = p_query_hash
                if simple_cse && haskey(plan_hash_result, p_query_hash)
                    alias_result[p_query.name] = plan_hash_result[p_query_hash]
                else
                    execute_query(alias_result, p_query, verbose)
                    plan_hash_result[p_query_hash] = alias_result[p_query.name]
                end
                total_exec_time += time() - exec_start
                if alias_result[p_query.name] isa Tensor && update_cards
                    count_start = time()
                    fix_cardinality!(alias_stats[p_query.name], count_non_default(alias_result[p_query.name]))
                    total_count_time += time() - count_start
                end
                phys_opt_start = time()
                condense_stats!(alias_stats[p_query.name]; cheap=false)
                total_phys_opt_time += time() - phys_opt_start
            end
        end
    end
    total_overall_time = time()-overall_start
    verbose >= 2 && println("Time to FAQ Opt: ", faq_opt_time)
    verbose >= 2 && println("Time to Split Opt: ", total_split_time)
    verbose >= 2 && println("Time to Phys Opt: ", total_phys_opt_time)
    verbose >= 1 && println("Time to Optimize: ", (faq_opt_time + total_split_time + total_phys_opt_time))
    verbose >= 1 && println("Time to Execute: ", total_exec_time)
    verbose >= 1 && println("Time to count: ", total_count_time)
    verbose >= 1 && println("Overall Time: ", total_overall_time)
    return (value=[alias_result[alias] for alias in output_aliases],
            opt_time=(faq_opt_time + total_split_time + total_phys_opt_time),
            execute_time= total_exec_time,
            overall_time=total_overall_time)
end

end
