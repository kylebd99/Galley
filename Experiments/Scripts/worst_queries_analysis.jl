using CSV
using DataFrames
using Statistics

greedy_data = CSV.read("Experiments/Results/human_true_greedy_DCStats_false_true.csv", DataFrame)
println("HUMAN ANALYSIS")
println("Min: $(minimum(greedy_data.Runtime))")
println("Median: $(median(greedy_data.Runtime))")
println("Max: $(maximum(greedy_data.Runtime))")
println("Mean: $(mean(greedy_data.Runtime))")
println("GeoMean: $(exp(mean(log.(greedy_data.Runtime))))")
for query in sort(filter((x)->x[:Runtime] >= quantile(greedy_data.Runtime, .9), greedy_data), [:Runtime], rev=true).QueryPath
    println(query)
end


greedy_data = CSV.read("Experiments/Results/yeast_lite_true_greedy_DCStats_false.csv", DataFrame)
println("YEAST ANALYSIS")
println("Min: $(minimum(greedy_data.Runtime))")
println("Median: $(median(greedy_data.Runtime))")
println("Max: $(maximum(greedy_data.Runtime))")
println("Mean: $(mean(greedy_data.Runtime))")
println("GeoMean: $(exp(mean(log.(greedy_data.Runtime))))")
for query in sort(filter((x)->x[:Runtime] >= quantile(greedy_data.Runtime, .9), greedy_data), [:Runtime], rev=true).QueryPath
    println(query)
end


greedy_data = CSV.read("Experiments/Results/hprd_lite_true_greedy_DCStats_false.csv", DataFrame)
println("HPRD_LITE ANALYSIS")
println("Min: $(minimum(greedy_data.Runtime))")
println("Median: $(median(greedy_data.Runtime))")
println("Max: $(maximum(greedy_data.Runtime))")
println("Mean: $(mean(greedy_data.Runtime))")
println("GeoMean: $(exp(mean(log.(greedy_data.Runtime))))")
for query in sort(filter((x)->x[:Runtime] >= quantile(greedy_data.Runtime, .9), greedy_data), [:Runtime], rev=true).QueryPath
    println(query)
end



greedy_data = CSV.read("Experiments/Results/aids_true_greedy_DCStats_false.csv", DataFrame)
println("AIDS ANALYSIS")
println("Min: $(minimum(greedy_data.Runtime))")
println("Median: $(median(greedy_data.Runtime))")
println("Max: $(maximum(greedy_data.Runtime))")
println("Mean: $(mean(greedy_data.Runtime))")
println("GeoMean: $(exp(mean(log.(greedy_data.Runtime))))")
for query in sort(filter((x)->x[:Runtime] >= quantile(greedy_data.Runtime, .9), greedy_data), [:Runtime], rev=true).QueryPath
    println(query)
end


greedy_data = CSV.read("Experiments/Results/dblp_lite_true_greedy_DCStats_false_true_5.csv", DataFrame)
println("DBLP_LITE ANALYSIS")
println("Min: $(minimum(greedy_data.Runtime))")
println("Median: $(median(greedy_data.Runtime))")
println("Max: $(maximum(greedy_data.Runtime))")
println("Mean: $(mean(greedy_data.Runtime))")
println("GeoMean: $(exp(mean(log.(greedy_data.Runtime))))")
for query in sort(filter((x)->x[:Runtime] >= quantile(greedy_data.Runtime, .9), greedy_data), [:Runtime], rev=true).QueryPath
    println(query)
end
