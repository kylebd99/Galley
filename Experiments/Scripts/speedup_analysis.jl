using CSV
using DataFrames
using Statistics
include("../Experiments.jl")
datasets = [human, aids, yeast_lite, dblp_lite]

for dataset in datasets
    greedy_exp = ExperimentParams(workload=dataset, faq_optimizer=greedy; stats_type=DCStats, max_kernel_size=5, warm_start=true, timeout=200)
    pruned_exp = ExperimentParams(workload=dataset, faq_optimizer=pruned; stats_type=DCStats, max_kernel_size=5, warm_start=true, timeout=200)
    greedy_data = CSV.read("Experiments/Results/$(param_to_results_filename(greedy_exp))", DataFrame)
    pruned_data = CSV.read("Experiments/Results/$(param_to_results_filename(pruned_exp))", DataFrame)

    joint_data = innerjoin(greedy_data, pruned_data, on=:QueryPath, makeunique=true)
    joint_data[!, :Speedup] = joint_data[!, :Runtime] ./ joint_data[!, :Runtime_1]
    println("$(uppercase(string(dataset))) ANALYSIS")
    println("Min: $(minimum(joint_data.Speedup))")
    println("Median: $(median(joint_data.Speedup))")
    println("Max: $(maximum(joint_data.Speedup))")
    println("Mean: $(mean(joint_data.Speedup))")
    println("GeoMean: $(exp(mean(log.(joint_data.Speedup))))")
    for query in sort(filter((x)->x[:Speedup] <= quantile(joint_data.Speedup, .05), joint_data), [:Speedup]).QueryPath
        println(query)
    end
end
