using SparseArrays
using Finch
using DelimitedFiles
using CSV, DataFrames
using DelimitedFiles: writedlm
using StatsPlots
using Plots
using Galley
using Galley: FAQ_OPTIMIZERS, fill_table, factor_to_table_name

include("experiment_params.jl")
include("subgraph_workload.jl")
include("load_workload.jl")
include("run_experiments.jl")
include("graph_results.jl")
include("finch_to_duckdb.jl")
include("CompetingMethods/faq_to_duckdb.jl")
