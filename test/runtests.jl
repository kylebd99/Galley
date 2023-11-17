using Test
using Finch
using SparseArrays
using Galley
using Galley: HyperTreeDecomposition, Bag, Factor, FAQInstance, decomposition_to_logical_plan, _recursive_insert_stats!
using Galley: expr_to_kernel, execute_tensor_kernel

# run subtests
#include("basic-tests.jl")
include("decomposition-tests.jl")
