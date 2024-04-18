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
