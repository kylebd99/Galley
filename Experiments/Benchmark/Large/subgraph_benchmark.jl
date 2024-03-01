include("../../Experiments.jl")


#datasets = instances(WORKLOAD)
datasets = [hprd_lite]

experiments = ExperimentParams[]
for dataset in datasets
#    push!(experiments, ExperimentParams(workload=dataset, faq_optimizer=naive))
#    push!(experiments, ExperimentParams(workload=dataset, faq_optimizer=hypertree_width))
#    push!(experiments, ExperimentParams(workload=dataset, faq_optimizer=greedy; stats_type=DCStats, warm_start=true, description="Ours"))
    push!(experiments, ExperimentParams(workload=dataset, faq_optimizer=greedy; stats_type=DCStats, use_duckdb=true, description="Ours + DuckDB"))
    push!(experiments, ExperimentParams(workload=dataset, faq_optimizer=naive; stats_type=NaiveStats, use_duckdb=true, description="DuckDB"))
#    push!(experiments, ExperimentParams(workload=dataset, faq_optimizer=greedy; stats_type=NaiveStats))
#    push!(experiments, ExperimentParams(workload=dataset, faq_optimizer=ordering))
end

#run_experiments(experiments)

graph_grouped_box_plot(experiments; y_type=overall_time, grouping=description, filename="3subgraph_counting_htd_comparison_overall")
graph_grouped_box_plot(experiments; y_type=opt_time, grouping=description, filename="3subgraph_counting_htd_comparison_opt")
graph_grouped_box_plot(experiments; y_type=execute_time, grouping=description, filename="3subgraph_counting_htd_comparison_execute")
