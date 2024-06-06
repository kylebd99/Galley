using CSV
using DataFrames
using Statistics


greedy_data = CSV.read("Experiments/Results/human_true_greedy_DCStats_false_true.csv", DataFrame)
pruned_data = CSV.read("Experiments/Results/human_true_pruned_DCStats_false_true.csv", DataFrame)
joint_data = innerjoin(greedy_data, pruned_data, on=:QueryPath, makeunique=true)
joint_data[!, :Speedup] = joint_data[!, :Runtime] ./ joint_data[!, :Runtime_1]
println("HUMAN ANALYSIS")
println("Min: $(minimum(joint_data.Speedup))")
println("Median: $(median(joint_data.Speedup))")
println("Max: $(maximum(joint_data.Speedup))")
println("Mean: $(mean(joint_data.Speedup))")
println("GeoMean: $(exp(mean(log.(joint_data.Speedup))))")
println(filter((x)->x[:Speedup] == minimum(joint_data.Speedup), joint_data).QueryPath)


greedy_data = CSV.read("Experiments/Results/yeast_lite_true_greedy_DCStats_false.csv", DataFrame)
pruned_data = CSV.read("Experiments/Results/yeast_lite_true_pruned_DCStats_false.csv", DataFrame)
joint_data = innerjoin(greedy_data, pruned_data, on=:QueryPath, makeunique=true)
joint_data[!, :Speedup] = joint_data[!, :Runtime] ./ joint_data[!, :Runtime_1]
println("YEAST ANALYSIS")
println("Min: $(minimum(joint_data.Speedup))")
println("Median: $(median(joint_data.Speedup))")
println("Max: $(maximum(joint_data.Speedup))")
println("Mean: $(mean(joint_data.Speedup))")
println("GeoMean: $(exp(mean(log.(joint_data.Speedup))))")
println(filter((x)->x[:Speedup] == minimum(joint_data.Speedup), joint_data).QueryPath)


greedy_data = CSV.read("Experiments/Results/hprd_lite_true_greedy_DCStats_false.csv", DataFrame)
pruned_data = CSV.read("Experiments/Results/hprd_lite_true_pruned_DCStats_false.csv", DataFrame)
joint_data = innerjoin(greedy_data, pruned_data, on=:QueryPath, makeunique=true)
joint_data[!, :Speedup] = joint_data[!, :Runtime] ./ joint_data[!, :Runtime_1]
println("HPRD_LITE ANALYSIS")
println("Min: $(minimum(joint_data.Speedup))")
println("Median: $(median(joint_data.Speedup))")
println("Max: $(maximum(joint_data.Speedup))")
println("Mean: $(mean(joint_data.Speedup))")
println("GeoMean: $(exp(mean(log.(joint_data.Speedup))))")
println(filter((x)->x[:Speedup] == minimum(joint_data.Speedup), joint_data).QueryPath)


greedy_data = CSV.read("Experiments/Results/aids_true_greedy_DCStats_false.csv", DataFrame)
pruned_data = CSV.read("Experiments/Results/aids_true_pruned_DCStats_false.csv", DataFrame)
joint_data = innerjoin(greedy_data, pruned_data, on=:QueryPath, makeunique=true)
joint_data[!, :Speedup] = joint_data[!, :Runtime] ./ joint_data[!, :Runtime_1]
println("AIDS ANALYSIS")
println("Min: $(minimum(joint_data.Speedup))")
println("Median: $(median(joint_data.Speedup))")
println("Max: $(maximum(joint_data.Speedup))")
println("Mean: $(mean(joint_data.Speedup))")
println("GeoMean: $(exp(mean(log.(joint_data.Speedup))))")
println(filter((x)->x[:Speedup] == minimum(joint_data.Speedup), joint_data).QueryPath)

greedy_data = CSV.read("Experiments/Results/dblp_lite_true_greedy_DCStats_false_true_5.csv", DataFrame)
pruned_data = CSV.read("Experiments/Results/dblp_lite_true_pruned_DCStats_false_true_5.csv", DataFrame)
joint_data = innerjoin(greedy_data, pruned_data, on=:QueryPath, makeunique=true)
joint_data[!, :Speedup] = joint_data[!, :Runtime] ./ joint_data[!, :Runtime_1]
println("DBLP ANALYSIS")
println("Min: $(minimum(joint_data.Speedup))")
println("Median: $(median(joint_data.Speedup))")
println("Max: $(maximum(joint_data.Speedup))")
println("Mean: $(mean(joint_data.Speedup))")
println("GeoMean: $(exp(mean(log.(joint_data.Speedup))))")
println(filter((x)->x[:Speedup] == minimum(joint_data.Speedup), joint_data).QueryPath)
