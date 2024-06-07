include("../Experiments.jl")
using Finch
using Galley
using Galley: t_dense, t_sparse_list, insert_statistics!

vertex_vectors, edge_matrices = load_subgraph_dataset(aids, DCStats, nothing)
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


f_times = []
f_r = nothing
for i in 1:5
    rank = Tensor(Dense(Element(0.0)), n)
    r = Tensor(Dense(Element(0.0)), n)
    @finch (r .= 0.0; for j=_; r[j] = 1.0/n end)
    damp = .85
    beta_score = (1 - damp)/n
    f_time = @elapsed begin
        @finch (rank .= 0; for j=_, i=_; rank[i] += scaled_edges[i, j] * r[j] end)
        @finch (r .= 0.0; for i=_; r[i] = beta_score + damp * rank[i] end)
        @finch (rank .= 0; for j=_, i=_; rank[i] += scaled_edges[i, j] * r[j] end)
        @finch (r .= 0.0; for i=_; r[i] = beta_score + damp * rank[i] end)
        @finch (rank .= 0; for j=_, i=_; rank[i] += scaled_edges[i, j] * r[j] end)
        @finch (r .= 0.0; for i=_; r[i] = beta_score + damp * rank[i] end)
        @finch (rank .= 0; for j=_, i=_; rank[i] += scaled_edges[i, j] * r[j] end)
        @finch (r .= 0.0; for i=_; r[i] = beta_score + damp * rank[i] end)
    end
    push!(f_times, f_time)
    global f_r = rank
end


out_degree_g = Materialize(t_dense, :j, Aggregate(+, :i, Input(edges, :i, :j)))
scaled_edges_g = Materialize(t_dense, t_sparse_list, :i, :j, MapJoin(ceil_div, Input(edges, :i, :j), Input(out_degree_g, :j)))
rank_1_g = Materialize(t_dense, :i, Aggregate(+, :j, MapJoin(*, Input(scaled_edges, :i, :j), Input(r, :j))))
r_1_g = Materialize(t_dense, :i, MapJoin(+, beta_score, MapJoin(*, Input(rank_1_g, :i), damp)))
rank_2_g = Materialize(t_dense, :i, Aggregate(+, :j, MapJoin(*, Input(scaled_edges, :i, :j), Input(r_1_g, :j))))
r_2_g = Materialize(t_dense, :i, MapJoin(+, beta_score, MapJoin(*, Input(rank_2_g, :i), damp)))
rank_3_g = Materialize(t_dense, :i, Aggregate(+, :j, MapJoin(*, Input(scaled_edges, :i, :j), Input(r_2_g, :j))))
r_3_g = Materialize(t_dense, :i, MapJoin(+, beta_score, MapJoin(*, Input(rank_3_g, :i), damp)))
rank_4_g = Materialize(t_dense, :i, Aggregate(+, :j, MapJoin(*, Input(scaled_edges, :i, :j), Input(r_3_g, :j))))
r_4_g = Materialize(t_dense, :i, MapJoin(+, beta_score, MapJoin(*, Input(rank_4_g, :i), damp)))
insert_statistics!(DCStats, r_4_g)
galley_times = []
galley_opt_times = []
result_galley = galley(Query(:out, r_4_g), ST=DCStats, simple_cse = false, verbose=3)
for i in 1:5
    local result_galley = galley(Query(:out, r_4_g), ST=DCStats, simple_cse = false, verbose=0)
    push!(galley_times, result_galley.execute_time)
    push!(galley_opt_times, result_galley.opt_time)
end

galley_cse_times = []
galley_cse_opt_times = []
result_galley = galley(Query(:out, r_4_g), ST=DCStats, verbose=3)
for i in 1:5
    local result_galley = galley(Query(:out, r_4_g), ST=DCStats, verbose=0)
    push!(galley_cse_times, result_galley.execute_time)
    push!(galley_cse_opt_times, result_galley.opt_time)
end
println("Galley Exec: $(minimum(galley_times))")
println("Galley Opt: $(minimum(galley_opt_times))")
println("Galley CSE Exec: $(minimum(galley_cse_times))")
println("Galley CSE Opt: $(minimum(galley_cse_opt_times))")
println("Finch Exec: $(minimum(f_times))")
println("F = G: $(all(abs.(f_r .- result_galley.value) .< .01))")
println("F = G: $(all(abs.(f_r .- result_galley.value) .< .01))")
