include("../Experiments.jl")


#datasets = instances(SUBGRAPH_DATASET)
datasets = [human]

experiments = ExperimentParams[]
for dataset in datasets
    push!(experiments, ExperimentParams(workload=dataset))
end

run_experiments(experiments)

graph_grouped_box_plot(experiments)
