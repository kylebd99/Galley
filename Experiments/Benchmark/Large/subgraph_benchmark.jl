include("../../Experiments.jl")

#datasets = [human, aids, yeast_lite, dblp_lite, youtube_lite]
datasets = [human, aids, yeast_lite, dblp_lite, youtube_lite]
experiments = ExperimentParams[]
for data in datasets
    push!(experiments, ExperimentParams(workload=data, faq_optimizer=pruned; stats_type=DCStats, warm_start=true, use_duckdb=false, description="Galley", timeout=600))
    push!(experiments, ExperimentParams(workload=data, faq_optimizer=greedy; stats_type=DCStats, warm_start=true, description="Galley (Greedy)", timeout=600))
#    push!(experiments, ExperimentParams(workload=data, faq_optimizer=pruned; stats_type=DCStats,  use_duckdb=true, description="Galley + DuckDB Backend", timeout=600))
#    push!(experiments, ExperimentParams(workload=data, faq_optimizer=naive; stats_type=NaiveStats, max_kernel_size=4, use_duckdb=true, description="DuckDB", timeout=600))
end

run_experiments(experiments; use_new_processes=true)
filename = "subgraph_counting_"
graph_grouped_box_plot(experiments; y_type=overall_time, y_lims=[10^-3, 10^3], grouping=description, group_order=["Galley", "Galley (Greedy)", "Galley + DuckDB Backend", "DuckDB"], filename="$(filename)overall1", y_label="Execute + Optimize Time (s)")
graph_grouped_bar_plot(experiments; y_type=opt_time, y_lims=[10^-3.2, 5], grouping=description, group_order=["Galley", "Galley (Greedy)", "Galley + DuckDB Backend", "DuckDB"],  filename="$(filename)opt1", y_label="Mean Optimization Time (s)")
graph_grouped_box_plot(experiments; y_type=execute_time, y_lims=[10^-4, 10^3], grouping=description, group_order=["Galley", "Galley (Greedy)", "Galley + DuckDB Backend", "DuckDB"],  filename="$(filename)execute1", y_label="Execution Time (s)")
graph_grouped_bar_plot(experiments; y_type=compile_time, y_lims=[10^-1, 10^1], grouping=description, group_order=["Galley", "Galley (Greedy)", "Galley + DuckDB Backend", "DuckDB"],  filename="$(filename)compile1", y_label="Mean Compile Time (s)")
