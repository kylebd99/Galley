# This file defines the physical plan language. This language should fully define the
# execution plan without any ambiguity.

abstract type TensorExpression end
TensorId = String

# The struct containing all information needed to compute a small tensor kernel using the finch compiler.
# This will be the output of the kernel optimizer
mutable struct TensorKernel
    kernel_root::PlanNode
    input_tensors::Dict{TensorId, Union{TensorKernel, Finch.Tensor, Number}}
    output_indices::Vector{IndexExpr}
    output_formats::Vector{LevelFormat}
    output_dims::Vector{Int}
    output_default::Any
    loop_order::Vector{IndexExpr}
end

function getFormatString(lf::LevelFormat)
    if lf == t_sparse_list
        return "sl"
    elseif lf == t_hash
        return "h"
    elseif lf == t_dense
        return "d"
    else
        return "[LIST LEVEL NEEDS FORMAT]"
    end
end

function printKernel(k::TensorKernel, verbosity)
    if verbosity <= 0
        return
    end
    printExpression(k.kernel_root)
    println()
    if verbosity <= 1
        return
    end
    println("Loop Order: $(k.loop_order)")
    if verbosity <= 2
        return
    end
    print("Output: [")
    prefix = ""
    for i in eachindex(k.output_indices)
        print(prefix)
        print("$(k.output_indices[i])::($(getFormatString(k.output_formats[i])), $(k.output_dims[i]))")
        prefix = ", "
    end
    println(" Def: $(k.output_default)]")
end
