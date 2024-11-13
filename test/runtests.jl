using Test
using Finch
using Finch: AsArray
using SparseArrays
using LinearAlgebra

using Galley
using Galley: t_sparse_list, t_dense
using Galley: canonicalize, insert_statistics!, get_reduce_query, AnnotatedQuery, reduce_idx!, cost_of_reduce, greedy_query_to_plan
using Galley: estimate_nnz, reduce_tensor_stats, condense_stats!, merge_tensor_stats

# run subtests
include("plan-tests.jl")
include("annotated-query-tests.jl")
include("basic-tests.jl")
include("stats-tests.jl")
include("finch-tests.jl")
#include("decomposition-tests.jl")
