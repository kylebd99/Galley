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
@enum SupportedAggregates t_custom_agg = -1 t_sum = 0 t_prod = 1 t_max_agg = 2 t_min_agg = 3
aggregateEnumToString::Dict{SupportedAggregates, String} = Dict(t_custom_agg=> "CustomAgg", t_sum => "Sum", t_prod=>"Prod", 
                                                                t_max_agg=>"MaxAgg",  t_min_agg=>"MinAgg")
struct AggregateExpr <: TensorExpression
    op::Any
    aggregate_type::SupportedAggregates
    aggregate_indices::Vector{String}
    input::TensorExpression
end
function Sum(input::TensorExpression, indices) return AggregateExpr(+, t_sum, indices, input) end
function Prod(input::TensorExpression, indices) return AggregateExpr(*, t_prod, indices, input) end
function MaxAgg(input::TensorExpression, indices) return AggregateExpr(max, t_max_agg, indices, input) end
function MinAgg(input::TensorExpression, indices) return AggregateExpr(min, t_min_agg, indices, input) end
function CustomAggregate(op, input, indices) return AggregateExpr(op, t_custom_agg, indices, input) end

# The operator struct refers to mathematical operations with a (generally small) number of inputs which does not rely on
# the size of domains, e.g. max(A[i],B[i],C[j]) or A[i]*B[i].
@enum SupportedOperators t_custom_op = -1 t_add = 0 t_mult = 1 t_max = 2 t_min = 3 
operatorEnumToString::Dict{SupportedOperators, String} = Dict(t_custom_op=>"custom_op", t_add => "Add", t_mult=>"Mult", t_max=>"Max", t_min=>"Min")
struct OperatorExpr <: TensorExpression
    op
    operator_type::SupportedOperators
    inputs::Vector{<:TensorExpression}  # Should we disallow aggregates within operators
end
function Add(inputs::Vector{<:TensorExpression}) return OperatorExpr(+, t_add, inputs) end
function Mult(inputs::Vector{<:TensorExpression}) return OperatorExpr(*, t_mult, inputs) end
function Max(inputs::Vector{<:TensorExpression}) return OperatorExpr(max, t_max, inputs) end
function Min(inputs::Vector{<:TensorExpression}) return OperatorExpr(min, t_min, inputs) end
function CustomOperator(op, inputs::Vector{<:TensorExpression}) return OperatorExpr(op, t_custom_op, inputs) end

function printExpression(exp::TensorExpression)
    if isa(exp, AggregateExpr)
        print(aggregateEnumToString[exp.aggregateType],"(")
        printExpression(exp.subExpression)
        print(";", exp.aggregateVariable, ")")
    elseif isa(exp, OperatorExpr)
        prefix = ""
        print(operatorEnumToString[exp.operatorType],"(")
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