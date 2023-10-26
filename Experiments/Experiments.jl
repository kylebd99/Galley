

using SparseArrays
using Finch
using Galley
using Galley: FAQ_OPTIMIZERS, naive, tree_decomp
using DelimitedFiles
using CSV, DataFrames
using DelimitedFiles: writedlm
using StatsPlots
using Plots

include("experiment_params.jl")
include("subgraph_workload.jl")
include("load_workload.jl")
include("run_experiments.jl")
include("graph_results.jl")
