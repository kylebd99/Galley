# This file defines an FAQ sub-problem which we will optimize using specialized methods.


struct Factor
    input::LogicalPlanNode
    variables::Vector{IndexExpr}
    stats::TensorStats

end

struct FAQInstance



end
