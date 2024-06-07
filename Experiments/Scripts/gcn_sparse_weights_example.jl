using Finch
using Finch: @finch_program_instance, Element, SparseListLevel, Dense, SparseHashLevel, SparseCOO, fsparse_impl
using Finch.FinchNotation: index_instance, variable_instance, tag_instance, literal_instance,
                        access_instance,  assign_instance, loop_instance, declare_instance,
                        block_instance, define_instance, call_instance, freeze_instance,
                        thaw_instance,
                        Updater, Reader, Dimensionless
using Galley
using Galley: t_dense, t_hash
using SparseArrays

n = 10000
k = 100
A = Tensor(Dense(SparseList(Element(0))), fsprand(Int, n, n, .1) .% 2)
W = Tensor(Dense(SparseList(Element(0))), fsprand(Int, k, k, .01)  .% 10)
h_0 = Tensor(Dense(Dense(Element(0))), rand(Int, k, n) .% 10)
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


@noinline function finch_instance_gnn(A, W, h_0)
    i_1 = Tensor(Dense(Dense(Element(0), k), n))
    i_1_write_access = access_instance(tag_instance(variable_instance(:i_1), i_1), literal_instance(Updater()), index_instance(:k2), index_instance(:n1))
    w_access = access_instance(tag_instance(variable_instance(:W), W), literal_instance(Reader()), index_instance(:k1), index_instance(:k2))
    h_0_access = access_instance(tag_instance(variable_instance(:h_0), h_0), literal_instance(Reader()), index_instance(:k1), index_instance(:n1))

    prgm1 = block_instance(declare_instance(tag_instance(variable_instance(:i_1), i_1), literal_instance(0)),
                        loop_instance(index_instance(:n1), Dimensionless(),
                        loop_instance(index_instance(:k2), Dimensionless(),
                        loop_instance(index_instance(:k1), Dimensionless(),
                        assign_instance(i_1_write_access, literal_instance(+), call_instance(literal_instance(*), w_access, h_0_access))
                            ))))
    Finch.execute(prgm1)

    h_2_no_max = Tensor(Dense(Dense(Element(0), k), n))
    h_2_no_max_access = access_instance(tag_instance(variable_instance(:h_2_no_max), h_2_no_max), literal_instance(Updater()), index_instance(:k2), index_instance(:n2))
    i_1_read_access = access_instance(tag_instance(variable_instance(:i_1), i_1), literal_instance(Reader()), index_instance(:k2), call_instance(literal_instance(follow), index_instance(:n1)))
    A_access = access_instance(tag_instance(variable_instance(:A), A), literal_instance(Reader()), call_instance(literal_instance(walk), index_instance(:n1)), index_instance(:n2))

    prgm2 = block_instance(declare_instance(tag_instance(variable_instance(:h_2_no_max), h_2_no_max), literal_instance(0)),
                        loop_instance(index_instance(:n2), Dimensionless(),
                        loop_instance(index_instance(:n1), Dimensionless(),
                        loop_instance(index_instance(:k2), Dimensionless(),
                        assign_instance(h_2_no_max_access, literal_instance(+), call_instance(literal_instance(*), A_access, i_1_read_access))
                            )))
    )
    Finch.execute(prgm2)
end

f_instance = @timed finch_instance_gnn(A, W, h_0)
f_instance = @timed finch_instance_gnn(A, W, h_0)

h_2_galley = Materialize(t_dense, t_dense, :k2, :n2,
                MapJoin(max, 0,
                    Aggregate(+, :n1, :k1,
                        MapJoin(*,  Input(A, :n1, :n2),
                                    Input(h_0, :k1, :n1),
                                    Input(W, :k1, :k2)))))
h_3_galley = MapJoin(max, 0,
                Aggregate(+, :n2, :k2,
                    MapJoin(*, Input(A, :n2, :n3),
                                Input(h_2_galley, :k2, :n2),
                                Input(W, :k2, :k3))))

two_hop_gnn_query = Query(:h_4, Materialize(t_dense, t_dense, :k3, :n3, h_3_galley))

h_3_galley = galley(two_hop_gnn_query, ST=DCStats, verbose=3)
h_3_galley = galley(two_hop_gnn_query, ST=DCStats, verbose=0)
println("Finch == Galley: ", t_2.value == h_3_galley.value)
println("Galley Opt & Execute: ", h_3_galley.opt_time, "   ", h_3_galley.execute_time)
println("Finch Execute: ", t_2.time)
println("Finch Instance: ", f_instance.time)


#println(sum(h_3))
#println(sum(h_3_galley.value))
#=
using DuckDB
dbconn = DBInterface.connect(DuckDB.DB, ":memory:")
h_3_duckdb = galley(deepcopy(two_hop_gnn_query), dbconn=dbconn, ST=DCStats, verbose=0)
println("DuckDB == Galley:", h_3_duckdb.value == h_3_galley.value)
println("DuckDB Opt & Execute: ", h_3_duckdb.opt_time, "   ", h_3_duckdb.execute_time)
 =#
