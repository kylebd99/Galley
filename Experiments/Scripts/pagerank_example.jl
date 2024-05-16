include("../Experiments.jl")
using Finch
using Galley
using Galley: t_dense, t_sparse_list

vertex_vectors, edge_matrices = load_subgraph_dataset(youtube, DCStats, nothing)
edges = Tensor(Dense(SparseList(Element(0))), edge_matrices[0].tns.val)
#edges = Tensor(Dense(SparseList(Element(0.0))), fsprand(Bool, 100, 100, .01))
n = size(edges)[1]

ceil_div(x,y) = y == 0.0 ? x : x/y

out_degree = Tensor(Dense(Element(0)))
@finch (out_degree .= 0; for j=_, i=_; out_degree[j] += edges[i, j] end)
scaled_edges = Tensor(Dense(SparseList(Element(0.0))))
@finch begin
    scaled_edges .= 0
    for j = _, i = _
        scaled_edges[i, j] = (edges[i, j] / out_degree[j]) * (edges[i,j] > 0)
    end
end
rank = Tensor(Dense(Element(0.0)), n)
r = Tensor(Dense(Element(0.0)), n)
@finch (r .= 0.0; for j=_; r[j] = 1.0/n end)
damp = .85
beta_score = (1 - damp)/n


out_degree_g = Materialize(t_dense, :j, Aggregate(+, :i, Input(edges, :i, :j)))
scaled_edges_g = Materialize(t_dense, t_sparse_list, :i, :j, MapJoin(ceil_div, Input(edges, :i, :j), Input(out_degree_g, :j)))
rank_1_g = Materialize(t_dense, :i, Aggregate(+, :j, MapJoin(*, Input(scaled_edges, :i, :j), Input(r, :j))))
r_1_g = Materialize(t_dense, :i, MapJoin(+, beta_score, MapJoin(*, Input(rank_1_g, :i), damp)))
rank_2_g = Materialize(t_dense, :i, Aggregate(+, :j, MapJoin(*, Input(scaled_edges, :i, :j), Input(r_1_g, :j))))
r_2_g = Materialize(t_dense, :i, MapJoin(+, beta_score, MapJoin(*, Input(rank_2_g, :i), damp)))

result_galley = galley(deepcopy(Query(:out, r_2_g)), ST=DCStats, verbose=3)
result_galley = galley(deepcopy(Query(:out, r_2_g)), ST=DCStats, verbose=0)
println("Galley Exec: $(result_galley.execute_time)")
println("Galley Opt: $(result_galley.opt_time)")


f_time = @elapsed begin
    @finch (rank .= 0; for j=_, i=_; rank[i] += scaled_edges[i, j] * r[j] end)
    @finch (r .= 0.0; for i=_; r[i] = beta_score + damp * rank[i] end)
    @finch (rank .= 0; for j=_, i=_; rank[i] += scaled_edges[i, j] * r[j] end)
    @finch (r .= 0.0; for i=_; r[i] = beta_score + damp * rank[i] end)
end
f_time = @elapsed begin
    @finch (rank .= 0; for j=_, i=_; rank[i] += scaled_edges[i, j] * r[j] end)
    @finch (r .= 0.0; for i=_; r[i] = beta_score + damp * rank[i] end)
    @finch (rank .= 0; for j=_, i=_; rank[i] += scaled_edges[i, j] * r[j] end)
    @finch (r .= 0.0; for i=_; r[i] = beta_score + damp * rank[i] end)
end

println("Finch Exec: $(f_time)")
println("F = G: $(all(abs.(r .- result_galley.value) .< .01))")

"""
    pagerank(adj; [nsteps], [damp])

Calculate `nsteps` steps of the page rank algorithm on the graph specified by
the adjacency matrix `adj`. `damp` is the damping factor.
"""
function pagerank(edges; nsteps=20, damp = 0.85)
    (n, m) = size(edges)
    @assert n == m
    out_degree = Tensor(Dense(Element(0)))
    @finch (out_degree .= 0; for j=_, i=_; out_degree[j] += edges[i, j] end)
    scaled_edges = Tensor(Dense(SparseList(Element(0.0))))
    @finch begin
        scaled_edges .= 0
        for j = _, i = _
            if out_degree[i] != 0
                scaled_edges[i, j] = edges[i, j] / out_degree[j]
            end
        end
    end
    r = Tensor(Dense(Element(0.0)), n)
    @finch (r .= 0.0; for j=_; r[j] = 1.0/n end)
    rank = Tensor(Dense(Element(0.0)), n)
    beta_score = (1 - damp)/n

    for step = 1:nsteps
        @finch (rank .= 0; for j=_, i=_; rank[i] += scaled_edges[i, j] * r[j] end)
        @finch (r .= 0.0; for i=_; r[i] = beta_score + damp * rank[i] end)
    end
    return r
end
