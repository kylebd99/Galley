# This file defines a prototype front-end which allows users to define tensor expressions and get their results.
module Galley

using AutoHashEquals
using Base: hash, copy, convert, getproperty, getfield, setfield!, getindex, ==, show
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
export Scalar, OutTensor, RenameIndices, declare_binary_operator, Σ, Mat, Agg
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

name_counter::UInt64 = 0

function galley_gensym(s::String)
    global name_counter += 1
    return Symbol(s*"_$name_counter")
end
galley_gensym(s::Symbol) = galley_gensym(string(s))

include("finch-algebra_ext.jl")
include("utility-funcs.jl")
include("PlanAST/PlanAST.jl")
include("TensorStats/TensorStats.jl")
include("FAQOptimizer/FAQOptimizer.jl")
include("PhysicalOptimizer/PhysicalOptimizer.jl")
include("ExecutionEngine/ExecutionEngine.jl")

# TODO:
#   - Convert a Finch HL query to a galley query
#   - On Finch Side:
#           - One query at a time to galley
#           - Isolate reformat_stats
#           - Fuse mapjoins & permutations

# Galley takes in a series of high level queries which define required outputs.
# Each query has the form:
#       Query(name, Materialize(formats..., indices..., expr))
# The inner expr can be any combination of MapJoin(op, args...) and Aggregate(op, idxs..., arg)
# with the leaves being Input(tns, idxs...), Alias(name, idxs...), or Value(v) where name refers
# to the results of a previous query.
function galley(input_queries::Vector{PlanNode};
                    faq_optimizer::FAQ_OPTIMIZERS=greedy,
                    ST=DCStats,
                    dbconn::Union{DuckDB.DB, Nothing}=nothing,
                    update_cards=true,
                    simple_cse=true,
                    max_kernel_size=10,
                    output_logical_plan=false,
                    output_physical_plan=false,
                    verbose=0)
    overall_start = time()
    # To avoid input corruption, we start by copying the input queries (except for the data)
    input_queries = map(plan_copy, input_queries)
    if verbose >= 1
        println("Input Queries : ")
        for input_query in input_queries
            println(input_query)
        end
    end
    # First, we perform high level optimization where each query is translated to one or
    # more queries with a simpler structure: Query(name, Aggregate(op, idxs, point_expr))
    # where point_expr is made up of just MapJoin, Input, and Alias nodes.
    opt_start, faq_opt_start = time(), time()
    logical_queries = []
    alias_stats, alias_hash = Dict{IndexExpr, TensorStats}(),  Dict{IndexExpr, UInt}()
    output_aliases = [input_query.name for input_query in input_queries]
    output_orders = Dict(input_query.name => input_query.expr.idx_order for input_query in input_queries)
    for input_query in input_queries
        logical_plan = high_level_optimize(faq_optimizer, input_query, ST, alias_stats, alias_hash, verbose)
        for query in logical_plan
            alias_hash[query.name.name] = cannonical_hash(query.expr, alias_hash)
            alias_stats[query.name.name] = query.expr.stats
        end
        append!(logical_queries, logical_plan)
    end
    faq_opt_time = time() - faq_opt_start
    if output_logical_plan
        return logical_queries
    end
    if verbose >= 1
        println("FAQ Opt Time: $faq_opt_time")
        println("--------------- Logical Plan ---------------")
        for query in logical_queries
            println(query)
        end
        println("--------------------------------------------")
    end

    # If we've been given a valid DBConn handle, we hand it off to DuckDB.
    if !isnothing(dbconn)
        return duckdb_execute_logical_plan(logical_queries,
                                            dbconn,
                                            only(output_aliases),
                                            output_orders[only(output_aliases)],
                                            time() - opt_start,
                                            verbose)
    end

    # We now compute the logical queries in order by performing the following steps:
    #   1. Query Splitting: We reduce queries to a size which is manageably compilable by Finch
    #   2. Physical Optimization: We make three decision about execution strategy for each query
    #           a. Loop Order (which also determines transpositions)
    #           b. Output Format
    #           c. Access Protocols
    #   3. Execution: No more decisions are made, we simply build the kernel and hand it to
    #      Finch.
    #   4. Touch Up: We check the actual output cardinality and fix our stats accordingly.
    total_split_time, total_phys_opt_time, total_exec_time, total_count_time = 0,0,0,0
    plan_hash_result, alias_result = Dict(), Dict{IndexExpr, Any}()
    for l_query in logical_queries
        split_start = time()
        split_queries = split_query(l_query, ST, max_kernel_size, alias_stats, verbose)
        total_split_time  += time() - split_start
        for s_query in split_queries
            phys_opt_start = time()
            physical_queries = logical_query_to_physical_queries(s_query, ST, alias_stats)
            total_phys_opt_time += time() - phys_opt_start
            for p_query in physical_queries
                verbose > 2 && println("--------------- Computing: $(p_query.name) ---------------")
                verbose > 2 && println(p_query)
                verbose > 3 && validate_physical_query(p_query)
                exec_start = time()
                p_query_hash = cannonical_hash(p_query.expr, alias_hash)
                alias_hash[p_query.name.name] = p_query_hash
                if simple_cse && haskey(plan_hash_result, p_query_hash)
                    alias_result[p_query.name.name] = plan_hash_result[p_query_hash]
                else
                    execute_query(alias_result, p_query, verbose)
                    plan_hash_result[p_query_hash] = alias_result[p_query.name.name]
                end
                total_exec_time += time() - exec_start
                if update_cards && alias_result[p_query.name.name] isa Tensor
                    count_start = time()
                    fix_cardinality!(alias_stats[p_query.name.name], count_non_default(alias_result[p_query.name.name]))
                    total_count_time += time() - count_start
                end
                phys_opt_start = time()
                condense_stats!(alias_stats[p_query.name.name]; cheap=false)
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
    return (value=[alias_result[alias.name] for alias in output_aliases],
            opt_time=(faq_opt_time + total_split_time + total_phys_opt_time + total_count_time),
            execute_time= total_exec_time,
            overall_time=total_overall_time)
end

function galley(input_query::PlanNode;
                    faq_optimizer::FAQ_OPTIMIZERS=greedy,
                    ST=DCStats,
                    dbconn::Union{DuckDB.DB, Nothing}=nothing,
                    update_cards=true,
                    simple_cse=true,
                    max_kernel_size=10,
                    verbose=0)
    result = galley(PlanNode[input_query];faq_optimizer=faq_optimizer,
                                ST=ST,
                                dbconn=dbconn,
                                update_cards=update_cards,
                                simple_cse=simple_cse,
                                max_kernel_size=max_kernel_size,
                                verbose=verbose)
    return (value=result.value[1],
            opt_time=result.opt_time,
            execute_time=result.execute_time,
            overall_time=result.overall_time)
end

end
