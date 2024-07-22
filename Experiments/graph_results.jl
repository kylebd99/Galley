# This is a good framework for graphing, but we need to make an experiment_params object first

@enum GROUP dataset faq_optimizer query_type stats_type description

@enum VALUE execute_time opt_time overall_time compile_time

function graph_grouped_box_plot(experiment_params_list::Vector{ExperimentParams};
                                        x_type::GROUP=dataset,
                                        y_type::VALUE=overall_time,
                                        grouping::GROUP=faq_optimizer,
                                        y_lims=[10^-5, 10^3],
                                        group_order = nothing,
                                        x_label=nothing,
                                        y_label=nothing,
                                        filename=nothing)
    # for now let's just use the dataset as the x-values and the cycle size as the groups
    x_values = []
    y_values = []
    groups = []
    for experiment_params in experiment_params_list
        # load the results
        results_path = "Experiments/Results/" * param_to_results_filename(experiment_params)
        # println("results path: ", results_path)
        results_df = CSV.read(results_path, DataFrame; normalizenames=true)
        meta_path = "Experiments/Results/" * param_to_results_filename(experiment_params; ext=".meta")
        meta_df = CSV.read(meta_path, DataFrame; normalizenames=true)
        num_attempted = only(meta_df.Attempted)
        num_completed = only(meta_df.Completed)
        for i in 1:(num_attempted-num_completed)
            push!(results_df, (string(experiment_params.workload), "", "", 600, sum(results_df.OptTime)/nrow(results_df), 0, String15(""), true), promote=true)
        end
        # get the x_value and grouping (same for all results in this experiment param)

        # keep track of the data points
        for i in 1:nrow(results_df)
            current_x = x_type == query_type ? results_df[i, :QueryType] : get_value_from_param(experiment_params, x_type)
            current_group = grouping == query_type ? results_df[i, :QueryType] : get_value_from_param(experiment_params, grouping)
            current_y = 0
            if y_type == execute_time
                current_y = results_df[i, :Runtime]
            elseif y_type == opt_time
                current_y = results_df[i, :OptTime]
            elseif y_type == compile_time
                current_y = results_df[i, :CompileTime]
            elseif y_type == overall_time
                current_y = results_df[i, :OptTime] + results_df[i, :Runtime]
            else # y_type == execute_time
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
    if !isnothing(group_order)
        groups = CategoricalArray(groups)
        levels!(groups, group_order)
    end
    # This seems to be necessary for using Plots.jl outside of the ipynb framework.
    # See this: https://discourse.julialang.org/t/deactivate-plot-display-to-avoid-need-for-x-server/19359/15
    ENV["GKSwstype"]="100"
    gbplot = groupedboxplot(x_values, y_values, group = groups, yscale =:log10,
                            ylims=y_lims, yticks=[10^-4, 10^-3, 10^-2, .1, 1, 10^1, 10^2, 10^3],
                            legend = :topleft, size = (1000, 600),
                            xtickfontsize=15,
                            ytickfontsize=15,
                            xguidefontsize=16,
                            yguidefontsize=16,
                            legendfontsize=16,
                            left_margin=10mm)
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
                                        group_order=nothing,
                                        filename=nothing)
    # for now let's just use the dataset as the x-values and the cycle size as the groups
    x_values = []
    y_values = Float64[]
    groups = []
    for experiment_params in experiment_params_list
        # load the results
        results_path = "Experiments/Results/" * param_to_results_filename(experiment_params)
        results_df = CSV.read(results_path, DataFrame; normalizenames=true)
        n_results = nrow(results_df)
        if y_type == opt_time
            results_df = combine(results_df, :OptTime=>sum, nrow, renamecols=false)
            results_df.OptTime = results_df.OptTime ./ n_results
        elseif y_type == compile_time
            results_df = combine(results_df, :CompileTime=>sum, nrow, renamecols=false)
            results_df.CompileTime = results_df.CompileTime ./ n_results
        end
        # keep track of the data points
        for i in 1:nrow(results_df)
            current_x = x_type == query_type ? results_df[i, :QueryType] : get_value_from_param(experiment_params, x_type)
            current_group = grouping == query_type ? results_df[i, :QueryType] : get_value_from_param(experiment_params, grouping)
            current_y = 0
            if y_type == opt_time
                current_y = results_df[i, :OptTime]
            elseif y_type == compile_time
                current_y = results_df[i, :CompileTime]
            elseif y_type == execute_time
                current_y = results_df[i, :Runtime]
            else # y_type == overall_time
                current_y = results_df[i, :OptTime] + results_df[i, :Runtime]
            end
            # push the errors and their groupings into the correct vector
            if current_y == 0
                continue
            end
            push!(x_values, current_x)
            push!(y_values, current_y)
            push!(groups, current_group)
        end
    end
    if !isnothing(group_order)
        groups = CategoricalArray(groups)
        levels!(groups, group_order)
    end
    results_filename = param_to_results_filename(experiment_params_list[1])
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
                            yscale =:log10,
                            yticks=[10^-4, 10^-3, 10^-2, .1, 1, 10^1, 10^2, 10^3],
                            ylims=y_lims,
                            legend = :topleft,
                            size = (1000, 600),
                            xtickfontsize=15,
                            ytickfontsize=15,
                            xguidefontsize=16,
                            yguidefontsize=16,
                            legendfontsize=16,
                            left_margin=10mm,
                            bottom_margin=10mm,
                            top_margin=10mm)
    if x_label !== nothing
        xlabel!(gbplot, x_label)
    else
        xlabel!(gbplot, string(x_type))
    end
    if y_label !== nothing
        ylabel!(gbplot, y_label)
    else
        ylabel!(gbplot, string(y_type))
    end
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
    elseif value_type == description
        return experiment_param.description
    else
        # default to grouping by faq_optimizer
        return experiment_param.faq_optimizer
    end
end
