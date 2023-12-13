# This is a good framework for graphing, but we need to make an experiment_params object first

@enum GROUP dataset faq_optimizer query_type stats_type

@enum VALUE runtime

function graph_grouped_box_plot(experiment_params_list::Vector{ExperimentParams};
                                        x_type::GROUP=dataset, y_type::VALUE=runtime,
                                        grouping::GROUP=faq_optimizer,
                                        x_label=nothing, y_label=nothing, filename=nothing)
    # for now let's just use the dataset as the x-values and the cycle size as the groups
    x_values = []
    y_values = []
    groups = []
    for experiment_params in experiment_params_list
        # load the results
        results_path = "Experiments/Results/" * param_to_results_filename(experiment_params)
        # println("results path: ", results_path)
        results_df = CSV.read(results_path, DataFrame; normalizenames=true)

        # get the x_value and grouping (same for all results in this experiment param)

        # keep track of the data points
        for i in 1:nrow(results_df)
            current_x = x_type == query_type ? results_df[i, :QueryType] : get_value_from_param(experiment_params, x_type)
            current_group = grouping == query_type ? results_df[i, :QueryType] : get_value_from_param(experiment_params, grouping)
            current_y = 0
            if y_type == runtime
                current_y = results_df[i, :Runtime]
            else # y_type == runtime
                current_y = results_df[i, :Runtime]
            end
            # push the errors and their groupings into the correct vector
            push!(x_values, current_x)
            push!(y_values, current_y)
            push!(groups, current_group)
        end
    end
    results_filename = param_to_results_filename(experiment_params_list[1])
    println("starting graphs")

    # This seems to be necessary for using Plots.jl outside of the ipynb framework.
    # See this: https://discourse.julialang.org/t/deactivate-plot-display-to-avoid-need-for-x-server/19359/15
    ENV["GKSwstype"]="100"
    gbplot = groupedboxplot(x_values, y_values, group = groups, yscale =:log10,
                            ylims=[10^-5, 10^1.5], yticks=[10^-4, 10^-3, 10^-2, .1, 1, 10^1],
                            legend = :outertopleft, size = (1000, 600))
    x_label !== nothing && xlabel!(gbplot, x_label)
    y_label !== nothing && ylabel!(gbplot, y_label)
    plotname = (isnothing(filename)) ? results_filename * ".png" : filename * ".png"
    savefig(gbplot, "Experiments/Figures/" * plotname)
end

function graph_grouped_bar_plot(experiment_params_list::Vector{ExperimentParams};
                                        x_type::GROUP=dataset,
                                        y_type::VALUE=estimate_error,
                                        grouping::GROUP=technique,
                                        x_label=nothing,
                                        y_label=nothing,
                                        y_lims=[0, 10],
                                        filename=nothing)
    # for now let's just use the dataset as the x-values and the cycle size as the groups
    x_values = []
    y_values = Float64[]
    groups = []
    for experiment_params in experiment_params_list
        # load the results
        results_path = "Experiments/Results/" * param_to_results_filename(experiment_params)
        results_df = CSV.read(results_path, DataFrame; normalizenames=true)

        # get the x_value and grouping (same for all results in this experiment param)
        println(results_df)
        # keep track of the data points
        for i in 1:nrow(results_df)
            current_x = x_type == query_type ? results_df[i, :QueryType] : get_value_from_param(experiment_params, x_type)
            current_group = grouping == query_type ? results_df[i, :QueryType] : get_value_from_param(experiment_params, grouping)
            current_y = 0
            if y_type == estimate_error
                current_y = results_df[i, :Estimate] / results_df[i, :TrueCard]
            elseif y_type == memory_footprint
                current_y = results_df[i, :MemoryFootprint]/(10^6)
            else
                     # y_type == runtime
                current_y = results_df[i, :EstimationTime]
            end
            # push the errors and their groupings into the correct vector
            push!(x_values, current_x)
            push!(y_values, current_y)
            push!(groups, current_group)
        end
    end
    results_filename = params_to_results_filename(experiment_params_list[1])
    println("starting graphs")

    # This seems to be necessary for using Plots.jl outside of the ipynb framework.
    # See this: https://discourse.julialang.org/t/deactivate-plot-display-to-avoid-need-for-x-server/19359/15
    ENV["GKSwstype"]="100"
    println(x_values)
    println(y_values)
    println(groups)
    gbplot = StatsPlots.groupedbar(x_values,
                            y_values,
                            group = groups,
#                            yscale =:log10,
                            ylims=y_lims,
                            legend = :outertopleft,
                            size = (1000, 600))
    x_label !== nothing && xlabel!(gbplot, x_label)
    y_label !== nothing && ylabel!(gbplot, y_label)
    plotname = (isnothing(filename)) ? results_filename * ".png" : filename * ".png"
    savefig(gbplot, "Experiments/Figures/" * plotname)
end




# default to grouping by dataset
function get_value_from_param(experiment_param::ExperimentParams, value_type::GROUP)
    if value_type == dataset
        return experiment_param.workload
    elseif value_type == faq_optimizer
        return experiment_param.faq_optimizer
    elseif value_type == stats_type
        return string(experiment_param.stats_type)
    else
        # default to grouping by faq_optimizer
        return experiment_param.faq_optimizer
    end
end
