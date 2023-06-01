using Finch

function initialize_tensor(formats::Vector{LevelFormat}, dims::Vector{Int64}, default_value)
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


function uniform_fiber(shape, sparsity; formats = [], default_value = 0)
    if formats == []
        formats = [t_sparse_list for _ in 1:length(shape)]
    end
    fiber = initialize_tensor(formats, shape, default_value)
    copyto!(fiber, fsprand(Bool, Tuple(shape), sparsity))
    return fiber
end

