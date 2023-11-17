function run_experiments(experiment_params::Vector{ExperimentParams})
    for experiment in experiment_params
        results = [("Workload", "QueryType", "QueryPath", "Runtime")]
        queries = load_workload(experiment.workload)
        if experiment.warm_start
            for query in queries
                println("Warm Start Query Path: ", query.query_path)
                galley(query.query; faq_optimizer = experiment.faq_optimizer)
            end
        end
        num_completed = 0
        num_attempted = length(queries)
        for query in queries
            println("Query Path: ", query.query_path)
            try
                result = @timed galley(query.query; faq_optimizer = experiment.faq_optimizer)
                push!(results, (string(experiment.workload), query.query_type, query.query_path, string(result.time)))
                num_completed += 1
            catch e
                println("Error Occurred: ", e)
            end
        end
        println("Attempted Queries: ", num_attempted)
        println("Completed Queries: ", num_completed)
        filename = "Experiments/Results/" * param_to_results_filename(experiment)
        writedlm(filename, results, ',')
    end
end
