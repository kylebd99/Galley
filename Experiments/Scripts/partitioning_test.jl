using Finch
include("../Experiments.jl")


vertices, edges = load_subgraph_dataset(aids, DCStats)

# E[i,j] * E[j,k] * E[k,l] * E[i, l]
function non_partitioned_4_cycle(edge_tensor)
    shape = size(edge_tensor)
    s = Finch.Scalar(0)
    c1 = Finch.Tensor(SparseHashLevel{1}(SparseHashLevel{1}(Element(0), (shape[1], )), (shape[2], )))
    edge_tensor_t = Finch.Tensor(SparseHashLevel{1}(SparseHashLevel{1}(Element(0), (shape[2], )), (shape[1], )))
    @finch begin
        for j=_,i=_
            edge_tensor_t[j, i] = edge_tensor[i,j]
        end
    end
    @finch begin
        c1 .= 0
        for j=_,  k=_, i=_
            c1[i, k] += edge_tensor[i, j] * edge_tensor_t[k, j]
        end
        for l=_, k=_, i=_
            s[] += c1[i, k] * edge_tensor[k, l] * edge_tensor[i, l]
        end
    end
    return s
end

# E[i,j] * E[j,k] * E[k,l] * E[i, l]

function partitioned_4_cycle(edge_tensor)
    shape = size(edge_tensor)
    s = Finch.Scalar(0)
    c1 = Finch.Tensor(Dense(Element(0), shape[1]))
    edge_tensor_t = Finch.Tensor(SparseHashLevel{1}(SparseHashLevel{1}(Element(0), (shape[2], )), (shape[1], )))

    println("Transposing Edge")
    @finch begin
        for j=_, i=_
            edge_tensor_t[j, i] = edge_tensor[i,j]
        end
    end
    println("Edge Transposed")
    @finch begin
        s .= 0
        for i=_
            c1 .= 0
            for j=_, k=_
                c1[k] += edge_tensor_t[j, i] * edge_tensor_t[k, j]
            end
            for l=_, k=_
                s[] += c1[k] * edge_tensor[k, l] * edge_tensor_t[l, i]
            end
        end
    end
    return s
end

#print(@timed non_partitioned_4_cycle(edges[0].args[2]))
print(@timed partitioned_4_cycle(edges[0].args[2]))
