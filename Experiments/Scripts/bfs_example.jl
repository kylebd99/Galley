include("../Experiments.jl")
using Finch
using Galley
using Galley: t_dense

vertex_vectors, edge_matrices = load_subgraph_dataset(dblp, DCStats, nothing)
A = Tensor(Dense(SparseList(Element(false))), edge_matrices[0].tns.val)
n = size(A)[1]
#n = 100000
#A = Tensor(Dense(SparseList(Pattern())), fsprand(Bool, n, n, 20 * (1.0/n)))
p = Tensor(SparseList(Pattern()), fsprand(Bool, n, 100 * (1.0/n)))
q = Tensor(SparseList(Pattern()), p)

p2 = Tensor(SparseList(Element(false)))
q2 = Tensor(SparseList(Element(false)))
p3 = Tensor(SparseList(Element(false)))
q3 = Tensor(SparseList(Element(false)))
p4 = Tensor(SparseList(Element(false)))
q4 = Tensor(SparseList(Element(false)))
f_time = @elapsed @finch begin
    q2 .= false
    for n2=_
        for n1=_
            q2[n2] <<choose(false)>>= ((q[n1] & A[n1, n2]) & !p[n2])
        end
    end

    p2 .= false
    for n1=_
        p2[n1] = q2[n1] | p[n1]
    end

    q3 .= false
    for n2=_
        for n1=_
            q3[n2] <<choose(false)>>= ((q2[n1] & A[n1, n2]) & !p2[n2])
        end
    end

    p3 .= false
    for n1=_
        p3[n1] = q3[n1] | p2[n1]
    end


    q4 .= false
    for n2=_
        for n1=_
            q4[n2] <<choose(false)>>= ((q3[n1] & A[n1, n2]) & !p3[n2])
        end
    end

    p4 .= false
    for n1=_
        p4[n1] = q4[n1] | p3[n1]
    end
end

f_time = @elapsed @finch begin
    q2 .= false
    for n2=_
        for n1=_
            q2[n2] <<choose(false)>>= ((q[n1] & A[n1, n2]) & !p[n2])
        end
    end

    p2 .= false
    for n1=_
        p2[n1] = q2[n1] | p[n1]
    end

    q3 .= false
    for n2=_
        for n1=_
            q3[n2] <<choose(false)>>= ((q2[n1] & A[n1, n2]) & !p2[n2])
        end
    end

    p3 .= false
    for n1=_
        p3[n1] = q3[n1] | p2[n1]
    end


    q4 .= false
    for n2=_
        for n1=_
            q4[n2] <<choose(false)>>= ((q3[n1] & A[n1, n2]) & !p3[n2])
        end
    end

    p4 .= false
    for n1=_
        p4[n1] = q4[n1] | p3[n1]
    end
end


q2 = Materialize(t_dense, :n2,
                    Aggregate(choose(false), :n1,
                        MapJoin(&,  Input(A, :n1, :n2),
                                    Input(q, :n1),
                                    MapJoin(!, Input(p, :n2)))))
p2 = Materialize(t_dense, :n2, MapJoin(|,
                                    Input(q2, :n2),
                                    Input(p, :n2)))

q3 = Materialize(t_dense, :n3,
                    Aggregate(choose(false), :n2,
                        MapJoin(&,  Input(A, :n2, :n3),
                                    Input(q2, :n2),
                                    MapJoin(!, Input(p2, :n3)))))
p3 = Materialize(t_dense, :n3, MapJoin(|,
                                    Input(q3, :n3),
                                    Input(p2, :n3)))

q4 = Materialize(t_dense, :n4,
                    Aggregate(choose(false), :n3,
                        MapJoin(&,  Input(A, :n3, :n4),
                                    Input(q3, :n3),
                                    MapJoin(!, Input(p3, :n4)))))
g_p4 = Materialize(t_dense, :n4, MapJoin(|,
                                    Input(q4, :n4),
                                    Input(p3, :n4)))

result = galley(deepcopy(Query(:out, g_p4)), ST=DCStats, verbose=3)
result_galley = galley(deepcopy(Query(:out, g_p4)), ST=DCStats, verbose=0)
println("Galley Exec: $(result_galley.execute_time)")
println("Galley Opt: $(result_galley.opt_time)")
println("Finch Exec: $(f_time)")
println("F = G: $(all(p4 .== result_galley.value))")

#=
using DuckDB
dbconn = DBInterface.connect(DuckDB.DB, ":memory:")
result_duckdb = galley(deepcopy(Query(:out, g_p4)), dbconn=dbconn, ST=DCStats, faq_optimizer=greedy, verbose=3)
println("DuckDB == Galley:", result_duckdb.value == result_galley.value)
println("DuckDB Opt & Execute: ", result_duckdb.opt_time, "   ", result_duckdb.execute_time)
 =#
