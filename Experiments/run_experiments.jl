function run_experiments(experiment_params::Vector{ExperimentParams})
    for experiment in experiment_params
        results = [("Workload", "QueryType", "QueryPath", "Runtime", "Result")]
        queries = load_workload(experiment.workload, experiment.stats_type)
        num_attempted = 0
        num_completed = 0
        num_correct = 0
        num_with_values = 0
        for query in queries
#            if any([occursin(x, query.query_type) for x in ["dense", "16", "24", "32"]])
#                continue
#            end
            println("Query Path: ", query.query_path)
            num_attempted +=1
            try
                if experiment.warm_start
                    println("Warm Start Query Path: ", query.query_path)
                    galley(query.query; faq_optimizer = experiment.faq_optimizer)
                end
                result = @timed galley(query.query; faq_optimizer = experiment.faq_optimizer, verbose=3)
                push!(results, (string(experiment.workload), query.query_type, query.query_path, string(result.time), string(result.value)))
                if !isnothing(query.expected_result)
                    if all(result.value .== query.expected_result)
                        num_correct += 1
                    end
                    num_with_values += 1
                end
                num_completed += 1
            catch e
                throw(e)
                println("Error Occurred: ", e)
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
