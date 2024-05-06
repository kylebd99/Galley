using Finch
using Galley
using SparseArrays

n = 10000
k = 100
A = Tensor(Dense(SparseList(Element(0))), fsprand(Int, n, n, .001) .% 100)
W = Tensor(Dense(Dense(Element(0))), rand(Int, k, k) .% 100)
h_0 = Tensor(Dense(Dense(Element(0))), rand(Int, k, n) .% 100)
nodes_of_interest = Tensor(SparseList(Element(0)), fsprand(Bool, n, .001))


h_1 = Tensor(Dense(Element(0.0)))
h_2 = Tensor(Dense(Dense(Element(0.0))))
t_1 = @elapsed @finch begin
    h_2 .= 0
    for k1 =_
        h_1 .= 0
        for n1=_
            for k2 =_
                h_1[n1] += h_0[k2, n1] * W[k2, k1]
            end
            h_1[n1] <<max>>= 0
        end
        for n1 = _
            for n2 = _
                h_2[k1, n1] += h_1[n2] * A[n2, n1]
            end
            h_2[k1, n1] <<max>>= 0
        end
    end
end

h_1 = Tensor(Dense(Dense(Element(0.0))))
h_3 = Tensor(Dense(Dense(Element(0.0))))
t_2 = @elapsed @finch begin
    h_1 .= 0
    for n1=_
        for k1 =_
            for k2 =_
                h_1[k1, n1] += h_0[k2, n1] * W[k2, k1]
            end
        end
    end

    h_3 .= 0
    for n1 = _
        for n2 = _
            for k = _
                h_3[k, n1] += h_1[k, n2] * A[n2, n1]
            end
        end
    end
    for n1=_
        for k1 = _
            h_3[k1, n1] <<max>>= 0
        end
    end
end

println(h_2 == h_3)
println(t_1)
println(t_2)

using Galley: t_dense, t_hash

function relu(x)
    return max(x, 0)
end

Finch.isidempotent(::Finch.AbstractAlgebra, ::typeof(relu)) = true
Finch.isassociative(::Finch.AbstractAlgebra, ::typeof(relu)) = true
Finch.iscommutative(::Finch.AbstractAlgebra, ::typeof(relu)) = true

h_1 = Materialize(t_dense, t_dense, :k1, :n1, MapJoin(max, 0, Aggregate(+,  :n2,
                                                                            :k2,
                                                                            MapJoin(*,  Input(A, :n2, :n1),
                                                                                        Input(h_0, :k2, :n2),
                                                                                        Input(W, :k2, :k1)))))
h_2 = MapJoin(max, 0, Aggregate(+, :n2, :k2, MapJoin(*, Input(A, :n2, :n1),
                                                        Input(h_1, :k2, :n2),
                                                        Input(W, :k2, :k1))))
println(h_2)
query = Query(:h_4, Materialize(t_dense, t_dense, :k1, :n1, h_2))
h_4 = galley(query, ST=NaiveStats, verbose=2)
println(h_4.opt_time, "   ", h_4.execute_time)

using DuckDB
dbconn = DBInterface.connect(DuckDB.DB, ":memory:")
h_4_duckdb = galley(query, dbconn=dbconn, ST=NaiveStats, verbose=2)
println(h_4_duckdb.value == h_4.value)
println(h_4_duckdb.opt_time, "   ", h_4_duckdb.execute_time)

#=
println(h_3 == h_4.value)
println(h_4.opt_time, "   ", h_4.execute_time)
h_1 =  Aggregate(+, :k2, MapJoin(*, Input(h_0, :k2, :n2), Input(W, :k2, :k1)))
h_5 = galley(Query(:h_4, Materialize(t_dense, t_dense, :k1, :n1,  Aggregate(+, :n2, MapJoin(*, Input(A, :n2, :n1), h_1)))), ST=DCStats, verbose=2)
println(h_5.opt_time, "   ", h_5.execute_time)

h_1 = MapJoin(*, Input(h_0, :k2, :n2), Input(W, :k2, :k1))
h_6 = galley(Query(:h_4, Materialize(t_dense, t_dense, :k1, :n1, MapJoin(relu, Aggregate(+, :n2, :k2, MapJoin(*, Input(A, :n2, :n1), h_1))))), ST=DCStats, verbose=2)
println(h_6.opt_time, "   ", h_6.execute_time)
 =#
