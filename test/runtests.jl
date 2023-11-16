using Test
using Finch
using SparseArrays
using Galley
using Galley: HyperTreeDecomposition, Bag, Factor, FAQInstance, decomposition_to_logical_plan, fill_in_stats
using Galley: expr_to_kernel, execute_tensor_kernel, insert_global_orders

# run subtests
#include("basic-tests.jl")
include("decomposition-tests.jl")
