using Test
using Finch
using SparseArrays
using DuckDB
using Galley


# run subtests
#include("basic-tests.jl")
include("decomposition-tests.jl")
include("stats-tests.jl")
