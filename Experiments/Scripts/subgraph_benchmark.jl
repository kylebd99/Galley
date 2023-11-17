include("../Experiments.jl")


#datasets = instances(WORKLOAD)
datasets = [aids]

experiments = ExperimentParams[]
for dataset in datasets
#    push!(experiments, ExperimentParams(workload=dataset, faq_optimizer=naive))
    push!(experiments, ExperimentParams(workload=dataset, faq_optimizer=hypertree_width))
    push!(experiments, ExperimentParams(workload=dataset, faq_optimizer=greedy))
end

run_experiments(experiments)

graph_grouped_box_plot(experiments; filename="subgraph_counting_htd_comparison")
