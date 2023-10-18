using Test
using Finch
using SparseArrays
using Galley
using Galley: HyperTreeDecomposition, Bag, Factor, FAQInstance, decomposition_to_logical_plan

# run subtests
include("basic-tests.jl")
include("decomposition-tests.jl")
