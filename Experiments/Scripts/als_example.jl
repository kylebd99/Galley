using Finch
using Galley
include("../Experiments.jl")

#vertex_vectors, edge_matrices = load_subgraph_dataset(yeast, DCStats, nothing)
#X = Tensor(Dense(SparseList(Element(0))), edge_matrices[0].tns.val)
#n = size(X)[1]
n = 10000
X = Tensor(Dense(SparseList(Element(0))), fsprand(Bool, n, n, 20 * (1.0/n)))
u = Tensor(Dense(Element(0)), rand(Int, n) .% 100)
v = Tensor(Dense(Element(0)), rand(Int, n) .% 100)
l = Scalar(0)

f_time = @elapsed @finch begin
    l.=0
    for i =_
        for j =_
            l[] += (X[j,i] - u[i]*v[j])^2
        end
    end
end
f_time = @elapsed @finch begin
    l.=0
    for i =_
        for j =_
            l[] += (X[j,i] - u[i]*v[j])^2
        end
    end
end

q = Materialize(Aggregate(+, :i, :j, MapJoin(*, MapJoin(+, Input(X, :j, :i), MapJoin(*, MapJoin(-, Input(u, :i)), Input(v, :j))),
                                                MapJoin(+, Input(X, :j, :i), MapJoin(*, MapJoin(-, Input(u, :i)), Input(v, :j))))))

result = galley(deepcopy(Query(:out, q)), ST=DCStats, verbose=3)
result_galley = galley(deepcopy(Query(:out, q)), ST=DCStats, verbose=0)
println("Galley Exec: $(result_galley.execute_time)")
println("Galley Opt: $(result_galley.opt_time)")
println("Finch Exec: $(f_time)")
println("F = G: $(l[] - result_galley.value))")
