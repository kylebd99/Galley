include("TensorQuery.jl")

# The set of allowed level formats provided by the Finch API
@enum LevelFormat t_sparse_list = 1 t_dense = 2

# The struct containing all information needed to compute a small tensor kernel using the finch compiler.
# This will be the output of the kernel optimizer
struct TensorKernel 
    kernel_root::TensorExpression

    input_tensors::Dict{TensorId, Finch.Fiber} 
    input_indices::Dict{TensorId, Vector{String}}
    input_protocols::Dict{TensorId, Vector{AccessProtocol}}
    
    output_indices::Vector{String}
    output_formats::Vector{LevelFormat}
    
    loop_order::Vector{String}
end