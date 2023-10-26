


function run_experiments(experiment_params::Vector{ExperimentParams})
    for experiment in experiment_params
        results = [("Workload", "QueryType", "QueryPath", "Runtime")]
        queries = load_workload(experiment.workload)
        if experiment.warm_start
            for query in queries
                galley(query.query)
            end
        end
        for query in queries
            println("Query Path: ", query.query_path)
            result = @timed galley(query.query)
            push!(results, (string(experiment.workload), query.query_type, query.query_path, string(result.time)))
        end
        filename = "Experiments/Results/" * param_to_results_filename(experiment)
        writedlm(filename, results, ',')
    end
end
