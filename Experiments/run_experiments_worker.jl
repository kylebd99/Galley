using SparseArrays
using Finch
using DelimitedFiles
using Distributed
using CSV, DataFrames
using DelimitedFiles: writedlm
using StatsPlots
using Plots
using DuckDB
using Galley
using Galley: FAQ_OPTIMIZERS, relabel_input, reindex_stats, fill_table
include("/local1/kdeeds/Galley/Experiments/experiment_params.jl")
include("/local1/kdeeds/Galley/Experiments/subgraph_workload.jl")
include("/local1/kdeeds/Galley/Experiments/load_workload.jl")


function clear_channel(c)
    while isready(c)
        take!(c)
    end
end

function attempt_experiment(experiment::ExperimentParams, starting_query, results_channel, status_channel)
    println("Starting Worker Experiment")
    dbconn = experiment.use_duckdb ? DBInterface.connect(DuckDB.DB, ":memory:") : nothing
    queries = load_workload(experiment.workload, experiment.stats_type, dbconn)
    num_attempted, num_completed, num_correct, num_with_values, _ = take!(status_channel)
    put!(status_channel, (num_attempted, num_completed, num_correct, num_with_values, false))
    for query in queries[starting_query:end]
        println("Query Path: ", query.query_path)
        if false && !occursin("Graph_12/uf_Q_0_1", query.query_path)
            continue
        end
        num_attempted +=1
        try
            if experiment.use_duckdb
                result = galley(query.query, ST=experiment.stats_type; faq_optimizer = experiment.faq_optimizer, dbconn=dbconn, verbose=0)
                result = galley(query.query, ST=experiment.stats_type; faq_optimizer = experiment.faq_optimizer, dbconn=dbconn, verbose=0)
                if result == "failed"
                    put!(results_channel, (string(experiment.workload), query.query_type, query.query_path, "0.0", "0.0", "0.0", string(true)))
                else
                    put!(results_channel, (string(experiment.workload), query.query_type, query.query_path, string(result.execute_time), string(result.opt_time), "0.0", string(result.value), string(false)))
                    if !isnothing(query.expected_result)
                        if result.value == query.expected_result
                            num_correct += 1
                        end
                        num_with_values += 1
                    end
                    num_completed += 1
                end
            else
                warm_start_time = 0
                if experiment.warm_start
                    println("Warm Start Query Path: ", query.query_path)
                    warm_start_time = @elapsed galley(query.query, ST=experiment.stats_type; faq_optimizer = experiment.faq_optimizer, update_cards=experiment.update_cards, simple_cse=experiment.simple_cse, max_kernel_size=experiment.max_kernel_size, verbose=0)
                    warm_start_time = @elapsed galley(query.query, ST=experiment.stats_type; faq_optimizer = experiment.faq_optimizer, update_cards=experiment.update_cards, simple_cse=experiment.simple_cse, max_kernel_size=experiment.max_kernel_size, verbose=3)
                    println("Warm Start Time: $warm_start_time")
                end
                result = galley(query.query, ST=experiment.stats_type; faq_optimizer = experiment.faq_optimizer, update_cards=experiment.update_cards, simple_cse=experiment.simple_cse, max_kernel_size=experiment.max_kernel_size, verbose=0)
                println(result)
                if result == "failed"
                    put!(results_channel, (string(experiment.workload), query.query_type, query.query_path, "0.0", "0.0", "0.0", "0.0", string(true)))
                else
                    put!(results_channel, (string(experiment.workload), query.query_type, query.query_path, string(result.execute_time), string(result.opt_time), string(warm_start_time - result.execute_time - result.opt_time), string(result.value), string(false)))
                    if !isnothing(query.expected_result)
                        if all(result.value .== query.expected_result)
                            num_correct += 1
                        else
                            println("Query Incorrect: $(query.query_path) Expected: $(query.expected_result) Returned: $(result.value)")
                            throw(Base.error("WRONG RESULT"))
                        end
                        num_with_values += 1
                    end
                    num_completed += 1
                end
            end

            clear_channel(status_channel)
            put!(status_channel, (num_attempted, num_completed, num_correct, num_with_values, false))
        catch e
            println("Error Occurred: ", e)
            throw(e)
        end
    end
    clear_channel(status_channel)
    put!(status_channel, (num_attempted, num_completed, num_correct, num_with_values, true))
end
