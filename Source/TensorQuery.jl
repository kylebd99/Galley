# This file defines the logical query described by the user. An example of this kind of query would be C[i][k] = \sum_j A[i][j]*B[j][k].
# The goal is to represent queries using a familar einstein notation-style format which builds complex expressions via composition.
# Notably, this query struct does not describe how the query will be executed.
#######################################################################################################################################

# This defines the list of access protocols allowed by the Finch API
@enum AccessProtocol t_walk = 1 t_fast_walk = 2 t_lead = 3 t_follow = 4 t_gallop = 5

abstract type TensorExpression end
TensorId = String

struct TensorAccess <: TensorExpression
    tensor_id::TensorId
    variables::Array{String}
    protocols::Array{AccessProtocol}
end

# The aggregate struct refers to mathematical operations which operate over a dimension of the tensors.
# These operations reduce the dimensionality of the result by removing that dimension.
@enum SupportedAggregates t_custom_agg = -1 t_sum = 0 t_prod = 1 t_max_agg = 2 t_min_agg = 3
aggregateEnumToString::Dict{SupportedAggregates, String} = Dict(t_custom_agg=> "custom_agg", t_sum => "Sum", t_prod=>"Prod", 
                                                                t_max_agg=>"MaxAgg",  t_min_agg=>"MinAgg")
struct AggregateExp <: TensorExpression
    op
    aggregate_type::SupportedAggregates
    aggregate_indices::Vector{String}
    input::TensorExpression
end
function Sum(input::TensorExpression, indices) return AggregateExp(+, t_sum, indices, input) end
function Prod(input::TensorExpression, indices) return AggregateExp(*, t_prod, indices, input) end
function MaxAgg(input::TensorExpression, indices) return AggregateExp(max, t_max_agg, indices, input) end
function MinAgg(input::TensorExpression, indices) return AggregateExp(min, t_min_agg, indices, input) end
function CustomAggregate(op, input, indices) return AggregateExp(op, t_custom_agg, indices, input) end

# The operator struct refers to mathematical operations with a (generally small) number of inputs which does not rely on
# the size of domains, e.g. max(A[i],B[i],C[j]) or A[i]*B[i].
@enum SupportedOperators t_custom_op = -1 t_add = 0 t_mult = 1 t_max = 2 t_min = 3 
operatorEnumToString::Dict{SupportedOperators, String} = Dict(t_custom_op=>"custom_op", t_add => "Add", t_mult=>"Mult", t_max=>"Max", t_min=>"Min")
struct OperatorExp <: TensorExpression
    op
    operator_type::SupportedOperators
    inputs::Vector{TensorExpression}  # Should we disallow aggregates within operators
end
function Add(inputs::Vector{<:TensorExpression}) return OperatorExp(+, t_add, inputs) end
function Mult(inputs::Vector{<:TensorExpression}) return OperatorExp(*, t_mult, inputs) end
function Max(inputs::Vector{<:TensorExpression}) return OperatorExp(max, t_max, inputs) end
function Min(inputs::Vector{<:TensorExpression}) return OperatorExp(min, t_min, inputs) end
function CustomOperator(op, inputs::Vector{<:TensorExpression}) return OperatorExp(op, t_custom_op, inputs) end

function printExpression(exp::TensorExpression)
    if isa(exp, AggregateExp)
        print(aggregateEnumToString[exp.aggregateType],"(")
        printExpression(exp.subExpression)
        print(";", exp.aggregateVariable, ")")
    elseif isa(exp, OperatorExp)
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