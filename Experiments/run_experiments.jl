macro timeout(seconds, expr_to_run, expr_when_fails)
    quote
        tsk = @task $(esc(expr_to_run))
        schedule(tsk)
        num_checks = $(esc(seconds))*10
        for i in 1:num_checks
            sleep(.1)
            if istaskdone(tsk)
                break
            end
        end
        if !istaskdone(tsk)
            schedule(tsk, InterruptException(), error=true)
        end
        try
            fetch(tsk)
        catch e
            throw(e)
            println("Error: $e")
            $(esc(expr_when_fails))
        end
    end
end

function run_experiments(experiment_params::Vector{ExperimentParams})
    for experiment in experiment_params
        results = [("Workload", "QueryType", "QueryPath", "Runtime", "OptTime", "CompileTime", "Result", "Failed")]
        dbconn = experiment.use_duckdb ? DBInterface.connect(DuckDB.DB, ":memory:") : nothing
        queries = load_workload(experiment.workload, experiment.stats_type, dbconn)
        num_attempted = 0
        num_completed = 0
        num_correct = 0
        num_with_values = 0
        for query in queries
            println("Query Path: ", query.query_path)#=
            if occursin("query_dense_4", query.query_path)
                continue
            end =#
            num_attempted +=1
            try
                if experiment.use_duckdb
                    result = @timeout experiment.timeout galley(query.query, ST=experiment.stats_type; faq_optimizer = experiment.faq_optimizer, dbconn=dbconn, verbose=0) "failed"
                    if result == "failed"
                        push!(results, (string(experiment.workload), query.query_type, query.query_path, "0.0", "0.0", "0.0", string(true)))
                    else
                        push!(results, (string(experiment.workload), query.query_type, query.query_path, string(result.execute_time), string(result.opt_time), "0.0", string(result.value), string(false)))
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
                        warm_start_time = @elapsed galley(query.query, ST=experiment.stats_type; faq_optimizer = experiment.faq_optimizer, verbose=4)
                        println("Warm Start Time: $warm_start_time")
                    end
                    result = galley(query.query, ST=experiment.stats_type; faq_optimizer = experiment.faq_optimizer, verbose=0)
                    if result == "failed"
                        push!(results, (string(experiment.workload), query.query_type, query.query_path, "0.0", "0.0", "0.0", "0.0", string(true)))
                    else
                        println(result)
                        push!(results, (string(experiment.workload), query.query_type, query.query_path, string(result.execute_time), string(result.opt_time), string(warm_start_time - result.execute_time - result.opt_time), string(result.value), string(false)))
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
            catch e
                println("Error Occurred: ", e)
                throw(e)
            end
        end
        println("Attempted Queries: ", num_attempted)
        println("Completed Queries: ", num_completed)
        println("Queries With Ground Truth: ", num_with_values)
        println("Correct Queries: ", num_correct)
        filename = "Experiments/Results/" * param_to_results_filename(experiment)
        writedlm(filename, results, ',')
    end
end
