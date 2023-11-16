using IterTools

@enum FAQ_OPTIMIZERS naive hypertree_width


include("faq-plan.jl")
include("faq-pruner.jl")
include("optimizers/naive-optimizer.jl")
include("optimizers/htw-optimizer.jl")
include("faq-optimizer.jl")
