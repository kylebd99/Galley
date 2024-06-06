include("../../Experiments.jl")


#datasets = instances(WORKLOAD)
datasets = [human, yeast_lite]
experiments = ExperimentParams[]
for data in datasets
    push!(experiments, ExperimentParams(workload=data, faq_optimizer=greedy; stats_type=DCStats, warm_start=true, description="Ours (greedy)", timeout=400))
    push!(experiments, ExperimentParams(workload=data, faq_optimizer=pruned; stats_type=DCStats, warm_start=true, description="Ours (pruned)", timeout=400))
    push!(experiments, ExperimentParams(workload=data, faq_optimizer=greedy; stats_type=DCStats, use_duckdb=true, description="Ours (greedy_cost) + DuckDB", timeout=400))
    push!(experiments, ExperimentParams(workload=data, faq_optimizer=pruned; stats_type=DCStats, use_duckdb=true, description="Ours (pruned) + DuckDB", timeout=400))
    push!(experiments, ExperimentParams(workload=data, faq_optimizer=naive; stats_type=NaiveStats, use_duckdb=true, description="DuckDB", timeout=400))
end
run_experiments(experiments)

graph_grouped_box_plot(experiments; y_type=overall_time, grouping=description, filename="subgraph_counting_overall")
graph_grouped_box_plot(experiments; y_type=opt_time, grouping=description, filename="subgraph_counting_opt")
graph_grouped_box_plot(experiments; y_type=execute_time, grouping=description, filename="subgraph_counting_execute")
graph_grouped_box_plot(experiments; y_type=compile_time, grouping=description, filename="subgraph_counting_compile")
