using Finch: Element, SparseList, Dense, SparseHashLevel, SparseCOO
using Random

function initialize_tensor(formats, dims::Vector{Int64}, default_value)
    B = Element(default_value)
    for i in range(1, length(dims))
        if formats[i] == t_sparse_list
            B = SparseList(B, dims[i])
        elseif formats[i] == t_dense
            B = Dense(B, dims[i])
        elseif formats[i] == t_hash
            B = SparseHashLevel(B, Tuple([dims[i]]))
        else
            println("Error: Attempted to initialize invalid level format type.")
        end
    end
    return Fiber!(B)
end


# Generates a fiber whose non-default entries are distributed uniformly randomly throughout.
function uniform_fiber(shape, sparsity; formats = [], default_value = 0, non_default_value = 1)
    if formats == []
        formats = [t_sparse_list for _ in 1:length(shape)]
    end
    fiber = initialize_tensor(formats, shape, default_value)
    I = Finch.fsprand_helper(Random.default_rng(), Tuple(shape), sparsity)
    V = [non_default_value for _ in 1:length(I[1])]
    data  = Fiber(SparseCOO{length(I), Tuple{map(eltype, I)...}}(Element{default_value}(V), Tuple(shape), I, [1, length(V) + 1]))
    copyto!(fiber, data)
    return fiber
end

# Call fsparse when constructing
