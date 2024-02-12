using IterTools

@enum FAQ_OPTIMIZERS naive hypertree_width greedy ordering

include("faq-plan.jl")
include("faq-utils.jl")
include("faq-pruner.jl")
include("optimizers/naive-optimizer.jl")
include("optimizers/htw-optimizer.jl")
include("optimizers/greedy-optimizer.jl")
include("optimizers/ordering-optimizer.jl")
include("faq-optimizer.jl")
