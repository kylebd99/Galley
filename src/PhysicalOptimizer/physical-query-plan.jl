# This file defines the physical plan language. This language should fully define the
# execution plan without any ambiguity.

TensorId = String

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

#=
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
=#
