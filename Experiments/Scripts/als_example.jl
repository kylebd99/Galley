using Finch
using Galley: insert_statistics!
using Galley
include("../Experiments.jl")

#vertex_vectors, edge_matrices = load_subgraph_dataset(yeast, DCStats, nothing)
#X = Tensor(Dense(SparseList(Element(0))), edge_matrices[0].tns.val)
#n = size(X)[1]
n = 10000
X = Tensor(Dense(SparseList(Element(0))), fsprand(Bool, n, n, 20))
u = Tensor(Dense(Element(0)), rand(Int, n) .% 100)
v = Tensor(Dense(Element(0)), rand(Int, n) .% 100)
f_l = nothing
f_times = []
for i in 1:5
    l = Scalar(0)
    f_time = @elapsed @finch begin
        l.=0
        for i =_
            for j =_
                l[] += (X[j,i] - u[i]*v[j])^2
            end
        end
    end
    push!(f_times, f_time)
    global f_l = l
end
q = Materialize(Σ(:i, :j, MapJoin(*, MapJoin(+, Input(X, :j, :i), MapJoin(*, MapJoin(-, Input(u, :i)), Input(v, :j))),
                                     MapJoin(+, Input(X, :j, :i), MapJoin(*, MapJoin(-, Input(u, :i)), Input(v, :j))))))
insert_statistics!(DCStats, q)
g_times = []
g_opt_times = []
for i in 1:5
    result_galley = galley([Query(:out, q)], ST=DCStats,  verbose=3)
    push!(g_times, result_galley.execute_time)
    push!(g_opt_times, result_galley.opt_time)
end
result = galley([Query(:out, q)], ST=DCStats, verbose=3)
println("Galley Exec: $(minimum(g_times))")
println("Galley Opt: $(minimum(g_opt_times))")
println("Finch Exec: $(minimum(f_times))")
println("F = G: $(f_l[] - result.value[1][])")
