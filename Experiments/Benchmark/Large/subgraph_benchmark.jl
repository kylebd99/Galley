include("../../Experiments.jl")


datasets = [youtube_lite]
experiments = ExperimentParams[]
for data in datasets
    push!(experiments, ExperimentParams(workload=data, faq_optimizer=greedy; stats_type=DCStats, warm_start=true, description="Ours (greedy)", timeout=600))
    push!(experiments, ExperimentParams(workload=data, faq_optimizer=pruned; stats_type=DCStats, warm_start=true, description="Ours (pruned)", timeout=600))
    push!(experiments, ExperimentParams(workload=data, faq_optimizer=greedy; stats_type=DCStats, use_duckdb=true, description="Ours (greedy_cost) + DuckDB", timeout=600))
    push!(experiments, ExperimentParams(workload=data, faq_optimizer=pruned; stats_type=DCStats, use_duckdb=true, description="Ours (pruned) + DuckDB", timeout=600))
    push!(experiments, ExperimentParams(workload=data, faq_optimizer=naive; stats_type=NaiveStats, use_duckdb=true, description="DuckDB", timeout=600))
end

run_experiments(experiments)

graph_grouped_box_plot(experiments; y_type=overall_time, y_lims=[10^-5, 10^3], grouping=description, filename="subgraph_counting_overall1")
graph_grouped_box_plot(experiments; y_type=opt_time, y_lims=[10^-5, 10^3], grouping=description, filename="subgraph_counting_opt1")
graph_grouped_box_plot(experiments; y_type=execute_time, y_lims=[10^-5, 10^3], grouping=description, filename="subgraph_counting_execute1")
graph_grouped_box_plot(experiments; y_type=compile_time, y_lims=[10^-5, 10^3], grouping=description, filename="subgraph_counting_compile1")
