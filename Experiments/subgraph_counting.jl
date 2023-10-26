

# In this function, we convert the property graph model to tensors. We do this by
# treating each node label as a 0/1 vector V_i indicating that a node has that label, and
# each edge label as a 0/1 matrix E_ij indicating whether there is an edge from i to j
# where the edge has the correct label

function load_dataset(path; subgraph_matching_data=false)
    n = 0
    edges::Dict{Tuple{Int, Int}, Int} = Dict()
    vertices::Dict{Int, Array{Int}} = Dict()
    edge_labels = []
    vertex_labels = []
    for line in eachline(path)
        if length(line) == 0
            continue
        end
        if line[1] == 'v'
            parts = split(line)
            labels = []
            if (subgraph_matching_data)
                push!(labels, parse(Int, parts[3]))
                vertices[parse(Int, parts[2])+1] =  labels
            else
                for x in parts[3:length(parts)]
                    push!(labels, parse(Int, x))
                end
                vertices[parse(Int,  parts[2]) + 1] =  labels
            end
            vertex_labels = union(vertex_labels, labels)
            n += 1
        elseif line[1] == 'e'
            parts = split(line)
            if (subgraph_matching_data)
                e1, e2 = parse(Int, parts[2])+1, parse(Int, parts[3])+1
                edges[(e1, e2)] = 0
                edges[(e2, e1)] = 0
                edge_labels = [0]
            else
                e1, e2, l1 = parse(Int, parts[2])+1, parse(Int, parts[3])+1, parse(Int, parts[4])
                edges[(e1, e2)] =  l1
            end
        end
    end
    edge_labels = union(collect(values(edges)))
    vertex_vectors = Dict()
    for label in vertex_labels
        node_ids= []
        for v in keys(vertices)
            if label in vertices[v]
                push!(node_ids, v)
            end
        end
        values = [1 for _ in node_ids]

        vertex_vector = Fiber!(SparseList(Element(0), n))
        copyto!(vertex_vector,  sparsevec(node_ids, values, n))
        vertex_vectors[label] = vertex_vector
    end

    edge_matrices = Dict()
    for label in edge_labels
        i_ids = []
        j_ids = []
        for edge in keys(edges)
            if label in edges[edge]
                push!(i_ids, edge[1])
                push!(j_ids, edge[2])
            end
        end
        values = [1 for _ in i_ids]

        edge_matrix = Fiber!(SparseList(SparseList(Element(0), n), n))
        copyto!(edge_matrix, sparse(i_ids, j_ids, values, n, n))
        edge_matrices[label] = edge_matrix
    end
    return vertex_vectors, edge_matrices
end

query_id_to_idx(i::Int) = IndexExpr("v_" * string(i))

function load_query(path, vertex_vectors, edge_matrices; subgraph_matching_data=false)
    n = 0
    query_edges::Dict{Tuple{Int, Int}, Int} = Dict()
    query_vertices::Dict{Int, Int} = Dict()
    for line in eachline(path)
        if length(line) == 0
            continue
        end
        if line[1] == 't'
            continue
        elseif line[1] == 'v'
            parts = split(line)
            if (subgraph_matching_data)
                data_label = -1
                label = parse(Int, parts[3])
                query_vertices[parse(Int, parts[2]) + 1] = label
            else
                data_label = parse(Int, parts[4])
                label = parse(Int, parts[3])
                query_vertices[parse(Int,  parts[2]) + 1] = label
            end
            n += 1
        elseif line[1] == 'e'
            parts = split(line)
            if (subgraph_matching_data)
                e1, e2 = parse(Int, parts[2])+1, parse(Int, parts[3])+1
                query_edges[(e1, e2)] = 0
            else
                e1, e2, l1 = parse(Int, parts[2]) + 1, parse(Int, parts[3]) + 1, parse(Int, parts[4])
                query_edges[(e1, e2)] =  l1
            end
        end
    end

    factors = []
    for v in keys(query_vertices)
        label = query_vertices[v]
        if label == -1
            continue
        end
        idx = query_id_to_idx(v)
        indices = Set([idx])
        vertex_tensor = InputTensor(vertex_vectors[label])[idx]
        vertex_factor = Factor(vertex_tensor, indices, indices, false, TensorStats([idx], vertex_vectors[label], nothing))
        push!(factors, vertex_factor)
    end

    for edge in keys(query_edges)
        label = query_edges[edge]
        if label == -1
            continue
        end
        l_idx = query_id_to_idx(edge[1])
        r_idx = query_id_to_idx(edge[2])
        indices = Set([l_idx, r_idx])
        edge_tensor = InputTensor(edge_matrices[label])[l_idx, r_idx]
        edge_factor = Factor(edge_tensor, indices, indices, false, TensorStats([l_idx, r_idx], edge_matrices[label], nothing))
        push!(factors, edge_factor)
    end
    faq = FAQInstance(*, +, Set(IndexExpr[]), Set([query_id_to_idx(v) for v in 1:n]), factors)
    return faq
end
