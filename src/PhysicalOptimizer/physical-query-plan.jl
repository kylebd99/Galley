# This file defines the physical plan language. This language should fully define the
# execution plan without any ambiguity.

abstract type TensorExpression end
TensorId = String

# This defines the list of access protocols allowed by the Finch API
@enum AccessProtocol t_walk = 1 t_lead = 2 t_follow = 3 t_gallop = 4 t_default = 5


mutable struct InputExpr <: TensorExpression
    tensor_id::TensorId
    input_indices::Vector{IndexExpr}
    input_protocols::Vector{AccessProtocol}
    stats::TensorStats
end

# The aggregate struct refers to mathematical operations which operate over a dimension of the tensors.
# These operations reduce the dimensionality of the result by removing the dimensions of
# `aggregate_indices`.
mutable struct AggregateExpr <: TensorExpression
    op::Any
    aggregate_indices::Set{IndexExpr}
    input::TensorExpression
end

# The operator struct refers to mathematical operations with a (generally small) number of inputs which does not rely on
# the size of domains, e.g. max(A[i],B[i],C[j]) or A[i]*B[i].
mutable struct OperatorExpr <: TensorExpression
    op
    inputs::Vector{<:TensorExpression}
end


# The reorder struct inserts a blocking operation which changes the input's
# layout/order of its indices.
mutable struct ReorderExpr <: TensorExpression
    index_order::Vector{IndexExpr}
    input::TensorExpression
end

function printExpression(exp::TensorExpression)
    if exp isa AggregateExpr
        print(exp.op,"(")
        printExpression(exp.input)
        print(";", exp.aggregate_indices, ")")
    elseif exp isa OperatorExpr
        prefix = ""
        print(exp.op,"(")
        for input in exp.inputs
            print(prefix)
            printExpression(input)
            prefix = ","
        end
        print(")")
    elseif exp isa ReorderExpr
        prefix = ""
        print("Reorder($(exp.index_order) ,")
        printExpression(exp.input)
        print(")")
    elseif exp isa InputExpr
        print(exp.tensor_id, "[")
        prefix = ""
        for i in 1:length(exp.input_indices)
            print(prefix)
            print(exp.input_indices[i], "::", exp.input_protocols[i])
            prefix = ","
        end
        print("]")
    end
end


# The struct containing all information needed to compute a small tensor kernel using the finch compiler.
# This will be the output of the kernel optimizer
mutable struct TensorKernel
    kernel_root::TensorExpression
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
