


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


function load_subgraph_dataset(dataset::WORKLOAD)
    if dataset == aids
        aids_data_file_path = "Experiments/Data/Subgraph_Data/aids/aids.txt"
        return load_dataset(aids_data_file_path)
    elseif dataset == human
        human_data_file_path = "Experiments/Data/Subgraph_Data/human/human.txt"
        return load_dataset(human_data_file_path)
    elseif dataset == lubm80
        lubm80_data_file_path = "Experiments/Data/Subgraph_Data/lubm80/lubm80.txt"
        return load_dataset(lubm80_data_file_path)
    elseif dataset == yago
        yago_data_file_path = "Experiments/Data/Subgraph_Data/yago/yago.txt"
        return load_dataset(yago_data_file_path)
    elseif dataset == yeast
        yeast_data_file_path = "Experiments/Data/Subgraph_Data/yeast/yeast.graph"
        return load_dataset(yeast_data_file_path, subgraph_matching_data=true)
    elseif dataset == hprd
        hprd_data_file_path = "Experiments/Data/Subgraph_Data/hprd/hprd.graph"
        return load_dataset(hprd_data_file_path, subgraph_matching_data=true)
    elseif dataset == wordnet
        wordnet_data_file_path = "Experiments/Data/Subgraph_Data/wordnet/wordnet.graph"
        return load_dataset(wordnet_data_file_path, subgraph_matching_data=true)
    elseif dataset == dblp
        dblp_data_file_path = "Experiments/Data/Subgraph_Data/dblp/dblp.graph"
        return load_dataset(dblp_data_file_path, subgraph_matching_data=true)
    elseif dataset == youtube
        youtube_data_file_path = "Experiments/Data/Subgraph_Data/youtube/youtube.graph"
        return load_dataset(youtube_data_file_path, subgraph_matching_data=true)
    elseif dataset == patents
        patents_data_file_path = "Experiments/Data/Subgraph_Data/patents/patents.graph"
        return load_dataset(patents_data_file_path, subgraph_matching_data=true)
    elseif dataset == eu2005
        eu2005_data_file_path = "Experiments/Data/Subgraph_Data/eu2005/eu2005.graph"
        return load_dataset(eu2005_data_file_path, subgraph_matching_data=true)
    end
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
        vertex_factor = Factor(vertex_tensor, indices, indices, false, TensorStats([idx], vertex_vectors[label]))
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
        edge_factor = Factor(edge_tensor, indices, indices, false, TensorStats([l_idx, r_idx], edge_matrices[label]))
        push!(factors, edge_factor)
    end
    faq = FAQInstance(*, +, Set{IndexExpr}(), Set([query_id_to_idx(v) for v in 1:n]), factors)
    return faq
end



function load_subgraph_workload(dataset::WORKLOAD)

    query_directories = Dict()
    query_directories[aids] = ["/aids/Chain_3/",
        "/aids/Chain_6/",
        "/aids/Chain_9/",
        "/aids/Chain_12/",
        "/aids/Cycle_3/",
        "/aids/Cycle_6/",
        "/aids/Flower_6/",
        "/aids/Flower_9/",
        "/aids/Flower_12/",
        "/aids/Graph_3/",
        "/aids/Graph_6/",
        "/aids/Graph_9/",
        "/aids/Graph_12/",
        "/aids/Petal_6/",
        "/aids/Petal_9/",
        "/aids/Petal_12/",
        "/aids/Star_3/",
        "/aids/Star_6/",
        "/aids/Star_9/",
        "/aids/Tree_3/",
        "/aids/Tree_6/",
        "/aids/Tree_9/",
        "/aids/Tree_12/"]
    query_directories[human] = ["/human/Chain_3/",
    "/human/Graph_3/",
    "/human/Star_3/",
    "/human/Tree_3/"]
    query_directories[lubm80] = ["/lubm80"]
    query_directories[yago] = ["/yago/Chain_3",
    "/yago/Chain_6",
    "/yago/Chain_9",
    "/yago/Chain_12",
    "/yago/Clique_6",
    "/yago/Clique_10",
    "/yago/Cycle_3",
    "/yago/Cycle_6",
    "/yago/Cycle_9",
    "/yago/Flower_6",
    "/yago/Graph_3",
    "/yago/Graph_6",
    "/yago/Graph_9",
    "/yago/Graph_12",
    "/yago/Petal_6",
    "/yago/Petal_9",
    "/yago/Petal_12",
    "/yago/Star_3",
    "/yago/Star_6",
    "/yago/Star_9",
    "/yago/Star_12",
    "/yago/Tree_3",
    "/yago/Tree_6",
    "/yago/Tree_9",
    "/yago/Tree_12",
    ]

    query_directories[yeast] = ["/yeast"]
    query_directories[hprd] = ["/hprd"]
    query_directories[wordnet] = ["/wordnet"]
    query_directories[youtube] = ["/youtube"]
    query_directories[patents] = ["/patents"]
    query_directories[eu2005] = ["/eu2005"]
    query_directories[dblp] = ["/dblp"]

    query_paths = [readdir("Experiments/Data/Subgraph_Queries" * dir, join=true) for dir in query_directories[dataset]]
    query_paths = [(query_paths...)...]

    vertex_vectors, edge_matrices = load_subgraph_dataset(dataset)
    all_queries = []
    println("Loading Queries For: ", dataset)
    for query_path in query_paths
        query = load_query(query_path, vertex_vectors, edge_matrices)
        query_type = ""
        if dataset == lubm80
            query_type = match(r".*/.*/lubm80_(.*).txt", query_path).captures[1]
        elseif IS_GCARE_DATASET[dataset]
            query_type = match(r".*/.*/(.*)_.*/.*", query_path).captures[1]
        else
            query_type = match(r".*/.*/query_(.*)_.*", query_path).captures[1]
        end
        push!(all_queries, (query=query, query_type=query_type, query_path=query_path))
    end
    return all_queries

end
