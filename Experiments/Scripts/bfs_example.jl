include("../Experiments.jl")
using Finch
using Galley
using Galley: t_dense, t_undef, insert_statistics!

vertex_vectors, edge_matrices = load_subgraph_dataset(youtube, DCStats, nothing)
A = Tensor(Dense(SparseList(Element(false))), edge_matrices[0].tns.val)
n = size(A)[1]
#n = 100000
#A = Tensor(Dense(SparseList(Pattern())), fsprand(Bool, n, n, 20 * (1.0/n)))
p = Tensor(SparseList(Element(false)), fsprand(Bool, n, 100))
q = Tensor(SparseList(Element(false)), p)

sparse_p4 = nothing
f_times = []
for i in 1:5
    p2 = Tensor(Sparse(Element(false)))
    q2 = Tensor(Sparse(Element(false)))
    p3 = Tensor(Sparse(Element(false)))
    q3 = Tensor(Sparse(Element(false)))
    p4 = Tensor(Sparse(Element(false)))
    q4 = Tensor(Sparse(Element(false)))
    f_time = @elapsed @finch begin
        q2 .= false
        for n2=_
            for n1=_
                q2[n1] <<choose(false)>>= ((q[n2] & A[n1, n2]) & !p[n1])
            end
        end

        p2 .= false
        for n1=_
            p2[n1] = q2[n1] | p[n1]
        end

        q3 .= false
        for n2=_
            for n1=_
                q3[n1] <<choose(false)>>= ((q2[n2] & A[n1, n2]) & !p2[n1])
            end
        end

        p3 .= false
        for n1=_
            p3[n1] = q3[n1] | p2[n1]
        end


        q4 .= false
        for n2=_
            for n1=_
                q4[n1] <<choose(false)>>= ((q3[n2] & A[n1, n2]) & !p3[n1])
            end
        end

        p4 .= false
        for n1=_
            p4[n1] = q4[n1] | p3[n1]
        end
    end
    push!(f_times, f_time)
    global sparse_p4 = p4
end

dense_p4 = nothing
f_dense_times = []
for i in  1:5
    p2 = Tensor(Dense(Element(false)))
    q2 = Tensor(Dense(Element(false)))
    p3 = Tensor(Dense(Element(false)))
    q3 = Tensor(Dense(Element(false)))
    p4 = Tensor(Dense(Element(false)))
    q4 = Tensor(Dense(Element(false)))
    f_dense_time = @elapsed begin
        @finch begin
            q2 .= false
            for n2=_
                for n1=_
                    q2[n1] <<choose(false)>>= ((q[n2] & A[n1, n2]) & !p[n1])
                end
            end
        end
        @finch begin
            p2 .= false
            for n1=_
                p2[n1] <<choose(false)>>= q2[n1] | p[n1]
            end
        end

        @finch begin
            q3 .= false
            for n2=_
                for n1=_
                    q3[n1] <<choose(false)>>= ((q2[n2] & A[n1, n2]) & !p2[n1])
                end
            end
        end

        @finch begin
            p3 .= false
            for n1=_
                p3[n1] <<choose(false)>>= q3[n1] | p2[n1]
            end
        end

        @finch begin
            q4 .= false
            for n2=_
                for n1=_
                    q4[n1] <<choose(false)>>=  ((q3[n2] & A[n1, n2]) & !p3[n1])
                end
            end
        end

        @finch begin
            p4 .= false
            for n1=_
                p4[n1] <<choose(false)>>= q4[n1] | p3[n1]
            end
        end
    end
    push!(f_dense_times, f_dense_time)
    global dense_p4 = p4
end

q2 = Materialize(t_dense, :n1,
                    Aggregate(choose(false), :n2,
                        MapJoin(&,  Input(A, :n1, :n2),
                                    Input(q, :n2),
                                    MapJoin(!, Input(p, :n1)))))
p2 = Materialize(t_dense, :n3, MapJoin(|,
                                    Input(q2, :n3),
                                    Input(p, :n3)))
q3 = Materialize(t_dense, :n4,
                    Aggregate(choose(false), :n5,
                        MapJoin(&,  Input(A, :n4, :n5),
                                    Input(q2, :n5),
                                    MapJoin(!, Input(p2, :n4)))))
p3 = Materialize(t_dense, :n6, MapJoin(|,
                                    Input(q3, :n6),
                                    Input(p2, :n6)))
q4 = Materialize(t_dense, :n7,
                    Aggregate(choose(false), :n8,
                        MapJoin(&,  Input(A, :n7, :n8),
                                    Input(q3, :n8),
                                    MapJoin(!, Input(p3, :n7)))))
g_p4 = Materialize(t_undef, :n9, MapJoin(|,
                                    Input(q4, :n9),
                                    Input(p3, :n9)))
insert_statistics!(DCStats, g_p4)
result_galley = galley(deepcopy(Query(:out, g_p4)), simple_cse=false, faq_optimizer=greedy, ST=DCStats, max_kernel_size=4, verbose=3)
execute_times = []
opt_times = []
for i in 1:5
    result_galley = galley(deepcopy(Query(:out, g_p4)), simple_cse=false, faq_optimizer=greedy, ST=DCStats, max_kernel_size=4, verbose=0)
    push!(execute_times, result_galley.execute_time)
    push!(opt_times, result_galley.opt_time)
end
result_galley_cse = galley(deepcopy(Query(:out, g_p4)), simple_cse=true, faq_optimizer=greedy, ST=DCStats, max_kernel_size=3, verbose=3)
execute_times_cse = []
opt_times_cse = []
for i in 1:5
    result_galley_cse = galley(deepcopy(Query(:out, g_p4)), simple_cse=true, faq_optimizer=greedy,ST=DCStats, max_kernel_size=3, verbose=0)
    push!(execute_times_cse, result_galley_cse.execute_time)
    push!(opt_times_cse, result_galley_cse.opt_time)
end
println("Galley (Non-CSE) Exec: $(minimum(execute_times))")
println("Galley (Non-CSE) Opt: $(minimum(opt_times))")
println("Galley (CSE) Exec: $(minimum(execute_times_cse))")
println("Galley (CSE) Opt: $(minimum(opt_times_cse))")
println("Finch Sparse Exec: $(minimum(f_times))")
println("Finch Dense Exec: $(minimum(f_dense_times))")
println("F = G: $(sum(sparse_p4) == sum(result_galley_cse.value))")
println("F = G: $(sum(dense_p4) == sum(result_galley_cse.value))")

#=
using DuckDB
dbconn = DBInterface.connect(DuckDB.DB, ":memory:")
result_duckdb = galley(deepcopy(Query(:out, g_p4)), dbconn=dbconn, ST=DCStats, faq_optimizer=greedy, verbose=3)
println("DuckDB == Galley:", result_duckdb.value == result_galley.value)
println("DuckDB Opt & Execute: ", result_duckdb.opt_time, "   ", result_duckdb.execute_time)
 =#
