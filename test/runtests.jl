using Test
using Finch
using SparseArrays
using DuckDB
using Galley
using Galley: t_sparse_list, t_dense


# run subtests
include("annotated-query-tests.jl")
include("basic-tests.jl")
include("stats-tests.jl")
#include("decomposition-tests.jl")
