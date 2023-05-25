# This file defines the physical plan language. This language should fully define the 
# execution plan without any ambiguity.

abstract type TensorExpression end
TensorId = String

# This defines the list of access protocols allowed by the Finch API
@enum AccessProtocol t_walk = 1 t_fast_walk = 2 t_lead = 3 t_follow = 4 t_gallop = 5

# The set of allowed level formats provided by the Finch API
@enum LevelFormat t_sparse_list = 1 t_dense = 2 t_hash = 3

struct InputExpr <: TensorExpression
    tensor_id::TensorId
    input_indices::Vector{String}
    input_protocols::Vector{AccessProtocol}
    stats::TensorStats
end

# The aggregate struct refers to mathematical operations which operate over a dimension of the tensors.
# These operations reduce the dimensionality of the result by removing that dimension.
struct AggregateExpr <: TensorExpression
    op::Any
    aggregate_indices::Vector{String}
    input::TensorExpression
end

# The operator struct refers to mathematical operations with a (generally small) number of inputs which does not rely on
# the size of domains, e.g. max(A[i],B[i],C[j]) or A[i]*B[i].
struct OperatorExpr <: TensorExpression
    op
    inputs::Vector{<:TensorExpression}
end

function printExpression(exp::TensorExpression)
    if isa(exp, AggregateExpr)
        print(exp.op,"(")
        printExpression(exp.subExpression)
        print(";", exp.aggregateVariable, ")")
    elseif isa(exp, OperatorExpr)
        prefix = ""
        print(exp.op,"(")
        for subExp in exp.inputs
            print(prefix)
            printExpression(subExp)
            prefix = ","
        end
        print(")")
    elseif isa(exp, Tensor)
        print(exp.TensorID, "(")
        prefix = ""
        for var in exp.variables
            print(prefix)
            print(var)
            prefix = ","
        end
        print(")")
    end
end


# The struct containing all information needed to compute a small tensor kernel using the finch compiler.
# This will be the output of the kernel optimizer
struct TensorKernel 
    kernel_root::TensorExpression
    stats::TensorStats
    input_tensors::Dict{TensorId, Union{TensorKernel, Finch.Fiber, Number}} 
    
    output_indices::Vector{String}
    output_formats::Vector{LevelFormat}
    
    loop_order::Vector{String}
end