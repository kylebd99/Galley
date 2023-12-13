include("../Experiments.jl")


#datasets = instances(WORKLOAD)
datasets = [aids]

experiments = ExperimentParams[]
for dataset in datasets
#    push!(experiments, ExperimentParams(workload=dataset, faq_optimizer=naive))
#    push!(experiments, ExperimentParams(workload=dataset, faq_optimizer=hypertree_width))
    push!(experiments, ExperimentParams(workload=dataset, faq_optimizer=greedy))
    push!(experiments, ExperimentParams(workload=dataset, faq_optimizer=greedy; stats_type=NaiveStats))
end

run_experiments(experiments)

graph_grouped_box_plot(experiments; grouping=stats_type, filename="subgraph_counting_htd_comparison")
