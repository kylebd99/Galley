include("../../Experiments.jl")


#datasets = instances(WORKLOAD)
datasets = [hprd_lite]

experiments = ExperimentParams[]
for dataset in datasets
#    push!(experiments, ExperimentParams(workload=dataset, faq_optimizer=naive))
#    push!(experiments, ExperimentParams(workload=dataset, faq_optimizer=hypertree_width))
    push!(experiments, ExperimentParams(workload=dataset, faq_optimizer=greedy; stats_type=DCStats, warm_start=true, description="Ours", timeout=500))
#    push!(experiments, ExperimentParams(workload=dataset, faq_optimizer=greedy; stats_type=DCStats, use_duckdb=true, description="Ours + DuckDB", timeout=500))
    push!(experiments, ExperimentParams(workload=dataset, faq_optimizer=naive; stats_type=NaiveStats, use_duckdb=true, description="DuckDB", timeout=500))
#    push!(experiments, ExperimentParams(workload=dataset, faq_optimizer=greedy; stats_type=NaiveStats))
#    push!(experiments, ExperimentParams(workload=dataset, faq_optimizer=ordering))
end

run_experiments(experiments)

graph_grouped_box_plot(experiments; y_type=overall_time, grouping=description, filename="hprd_lite_subgraph_counting_overall")
graph_grouped_box_plot(experiments; y_type=opt_time, grouping=description, filename="hprd_lite_subgraph_counting_opt")
graph_grouped_box_plot(experiments; y_type=execute_time, grouping=description, filename="hprd_lite_subgraph_counting_execute")
graph_grouped_box_plot(experiments; y_type=compile_time, grouping=description, filename="hprd_lite_subgraph_counting_compile")
