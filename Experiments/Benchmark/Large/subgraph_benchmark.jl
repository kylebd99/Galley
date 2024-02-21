include("../../Experiments.jl")


#datasets = instances(WORKLOAD)
datasets = [aids]

experiments = ExperimentParams[]
for dataset in datasets
#    push!(experiments, ExperimentParams(workload=dataset, faq_optimizer=naive))
#    push!(experiments, ExperimentParams(workload=dataset, faq_optimizer=hypertree_width))
    push!(experiments, ExperimentParams(workload=dataset, faq_optimizer=greedy; stats_type=DCStats, warm_start=true, description="Ours"))
#    push!(experiments, ExperimentParams(workload=dataset, faq_optimizer=greedy; stats_type=NaiveStats))
    push!(experiments, ExperimentParams(workload=dataset, faq_optimizer=greedy; stats_type=NaiveStats, use_duckdb=true, description="DuckDB"))
#    push!(experiments, ExperimentParams(workload=dataset, faq_optimizer=ordering))
end

#run_experiments(experiments)

graph_grouped_box_plot(experiments; grouping=description, filename="subgraph_counting_htd_comparison 2")
