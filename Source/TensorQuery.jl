# This file defines the logical query described by the user. An example of this kind of query would be C[i][k] = \sum_j A[i][j]*B[j][k].
# The goal is to represent queries using a familar einstein notation-style format which builds complex expressions via composition.
# Notably, this query struct does not describe how the query will be executed.
#######################################################################################################################################

abstract type Expression end

struct Tensor <: Expression
    name::String
    variables::Array{String}
end

# The aggregate struct refers to mathematical operations which operate over a dimension of the tensors.
# These operations reduce the dimensionality of the result by removing that dimension.
@enum supportedAggregates summation = 0 product = 1 maximumAggregate = 2 minimumAggregate = 3
aggregateEnumToString::Dict{supportedAggregates, String} = Dict(summation => "Sum", product=>"Prod", 
                                                                 maximumAggregate=>"MaxAgg",  minimumAggregate=>"MinAgg")
struct AggregateExp <: Expression
    aggregateType::supportedAggregates
    subExpression::Expression
    aggregateVariable::String
end
function Sum(input, var) return AggregateExp(summation, input, var) end
function Prod(input, var) return AggregateExp(product, input, var) end
function MaxAgg(input, var) return AggregateExp(maximumAggregate, input, var) end
function MinAgg(input, var) return AggregateExp(minimumAggregate, input, var) end

# The operator struct refers to mathematical operations with a (generally small) number of inputs which does not rely on
# the size of domains, e.g. max(A[i],B[i],C[j]) or A[i]*B[i].
@enum supportedOperators addition = 0 multiplication = 1 maximum = 2 minimum = 3
operatorEnumToString::Dict{supportedOperators, String} = Dict(addition => "Add", multiplication=>"Mult", maximum=>"Max", minimum=>"Min")
struct OperatorExp <: Expression
    operatorType::supportedOperators
    inputs::Array{Expression}  # Should we disallow aggregates within operators
end
function Add(input) return OperatorExp(addition, input) end
function Mult(input) return OperatorExp(multiplication, input) end
function Max(input) return OperatorExp(maximum, input) end
function Min(input) return OperatorExp(minimum, input) end

struct Query
    headName::String
    rightHandSide::Expression
end

function printExpression(exp::Expression)
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
        print(exp.name, "(")
        prefix = ""
        for var in exp.variables
            print(prefix)
            print(var)
            prefix = ","
        end
        print(")")
    end
end

function printQuery(query::Query)
    print(query.headName, "=")
    printExpression(query.rightHandSide)
end

# Check if a query is supported by our engine.
function queryIsSupported(query::Query)
    println("queryIsSupported Not Implemented")
    return true
end
