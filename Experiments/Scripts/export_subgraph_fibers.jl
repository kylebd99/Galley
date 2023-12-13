include("../Experiments.jl")

#for dataset in [aids, dblp, eu2005, hprd, human, lubm80, patents, wordnet, yago, yeast, youtube]
for dataset in [yago]
    vertex_vecs, edge_matrices = load_subgraph_dataset(dataset)
    dataset_directory = "DatasetExport/"*string(dataset)*"/"
    for (l, v) in vertex_vecs
        output_file = dataset_directory * string(dataset) * "_V_" * string(l) *".bsp.h5"
        Finch.fwrite(output_file, v)
    end

    for (l, e) in edge_matrices
        output_file = dataset_directory * string(dataset) * "_E_" * string(l) *".bsp.h5"
        Finch.fwrite(output_file, e)
    end
end
