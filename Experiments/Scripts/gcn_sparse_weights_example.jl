using Finch
using Galley
using Galley: t_dense, t_hash, insert_statistics!
using SparseArrays

n = 100000
k = 1000
A = Tensor(Dense(SparseList(Element(0))), fsprand(Int, n, n, 500000) .% 2)
W = Tensor(Dense(SparseList(Element(0))), fsprand(Int, k, k, 10000)  .% 10)
h_0 = Tensor(Dense(SparseList(Element(0))), fsprand(Int, k, n, 1000000) .% 10)
# nodes_of_interest = Tensor(SparseList(Element(0)), fsprand(Int, n, .001).% 2)

@noinline function finch_gnn(A, W, h_0)
    i_1 = Tensor(Dense(Dense(Element(0))))
    h_2_no_max = Tensor(Dense(Dense(Element(0))))
    h_2 = Tensor(Dense(Dense(Element(0))))
    i_2 = Tensor(Dense(Dense(Element(0))))
    h_3_no_max = Tensor(Dense(Dense(Element(0))))
    h_3 = Tensor(Dense(Dense(Element(0))))
    @finch begin
        i_1 .= 0
        for n1=_
            for k2 =_
                for k1 =_
                    i_1[k2, n1] += h_0[k1, n1] * W[k1, k2]
                end
            end
        end
    end

    @finch begin
        h_2_no_max .= 0
        for n2 = _
            for n1 = _
                for k2 = _
                    h_2_no_max[k2, n2] += i_1[k2, n1] * A[n1, n2]
                end
            end
        end
    end

    @finch begin
        h_2 .= 0
        for n2=_
            for k2 = _
                h_2[k2, n2] = max(h_2_no_max[k2, n2], 0)
            end
        end
    end

    @finch begin
        i_2 .= 0
        for n2=_
            for k3 =_
                for k2 =_
                    i_2[k3, n2] += h_2[k2, n2] * W[k2, k3]
                end
            end
        end
    end

    @finch begin
        h_3_no_max .= 0
        for n3 = _
            for n2 = _
                for k3 = _
                    h_3_no_max[k3, n3] += i_2[k3, n2] * A[n2, n3]
                end
            end
        end
    end

    @finch begin
        h_3 .= 0
        for n3=_
            for k3 = _
                h_3[k3, n3] = max(h_3_no_max[k3,n3], 0)
            end
        end
    end
    return h_3
end

t_2 = @timed finch_gnn(A, W, h_0)
t_2 = @timed finch_gnn(A, W, h_0)

h_2_galley = Materialize(t_dense, t_dense, :k2, :n2,
                MapJoin(max, 0,
                    Aggregate(+, :n1, :k1,
                        MapJoin(*,  Input(A, :n1, :n2),
                                    Input(h_0, :k1, :n1),
                                    Input(W, :k1, :k2)))))
h_3_galley = MapJoin(max, 0,
                Aggregate(+, :n2, :k2,
                    MapJoin(*, Input(A, :n2, :n3),
                                h_2_galley[:k2, :n2],
                                Input(W, :k2, :k3))))

two_hop_gnn_query = Query(:h_4, Materialize(t_dense, t_dense, :k3, :n3, h_3_galley))
insert_statistics!(DCStats, two_hop_gnn_query)

h_3_galley = galley([two_hop_gnn_query], ST=DCStats, faq_optimizer=pruned, verbose=3)
h_3_galley = galley([two_hop_gnn_query], ST=DCStats, faq_optimizer=pruned, verbose=0)
println("Finch == Galley: ", t_2.value == h_3_galley.value[1])
println("Galley Opt & Execute: ", h_3_galley.opt_time, "   ", h_3_galley.execute_time)
println("Finch Execute: ", t_2.time)

#= using DuckDB
dbconn = DBInterface.connect(DuckDB.DB, ":memory:")
h_3_duckdb = galley(deepcopy(two_hop_gnn_query), dbconn=dbconn, ST=DCStats, verbose=0)
println("DuckDB == Galley:", h_3_duckdb.value == h_3_galley.value)
println("DuckDB Opt & Execute: ", h_3_duckdb.opt_time, "   ", h_3_duckdb.execute_time)
 =#
