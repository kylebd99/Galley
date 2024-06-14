include("../Experiments.jl")
using Finch
using Galley
using Galley: t_dense, t_undef, insert_statistics!
using MatrixDepot
using SparseArrays

matrix = "GAP/GAP-road"
println("Running BFS for $matrix")
println("Loading Data: $matrix")
A_int = Tensor(Dense(SparseList(Element(0))), SparseMatrixCSC(matrixdepot(matrix)))
A = Tensor(Dense(SparseList(Element(false))))
@finch (A .= 0; for i=_ for j =_ A[j,i] = A_int[j,i] != 0 end end )
n = size(A)[1]
p = Tensor(SparseList(Element(false)), fsprand(Bool, n, 10))
q = Tensor(SparseList(Element(false)), p)

n_rounds = 15
println("Running Sparse Version: $matrix")
sparse_p4 = nothing
f_times = []
for i in 1:3
    p1 = Tensor(SparseDict(Element(false)), p)
    q1 = Tensor(SparseDict(Element(false)), q)
    p2 = Tensor(SparseDict(Element(false)))
    q2 = Tensor(SparseDict(Element(false)))
    p3 = Tensor(SparseDict(Element(false)))
    q3 = Tensor(SparseDict(Element(false)))
    p4 = Tensor(SparseDict(Element(false)))
    q4 = Tensor(SparseDict(Element(false)))
    rounds = 0
    f_time = 0
    while (rounds == 0 || sum(q4) > 0) && rounds < n_rounds
        f_time += @elapsed @finch begin
            q2 .= false
            for n2=_
                for n1=_
                    q2[n1] <<choose(false)>>= ((q1[n2] & A[n1, n2]) & !p1[n1])
                end
            end

            p2 .= false
            for n1=_
                p2[n1] = q2[n1] | p1[n1]
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
        p1 = Tensor(SparseDict(Element(false)), p4)
        q1 = Tensor(SparseDict(Element(false)), q4)
        rounds += 1
    end
    push!(f_times, f_time)
    global sparse_p4 = p4
end

println("Running Dense Version: $matrix")
dense_p4 = nothing
f_dense_times = []
for i in  1:3
    p1 = Tensor(Dense(Element(false)), p)
    q1 = Tensor(Dense(Element(false)), q)
    p2 = Tensor(Dense(Element(false)))
    q2 = Tensor(Dense(Element(false)))
    p3 = Tensor(Dense(Element(false)))
    q3 = Tensor(Dense(Element(false)))
    p4 = Tensor(Dense(Element(false)))
    q4 = Tensor(Dense(Element(false)))
    rounds = 0
    f_dense_time = 0
    while  (rounds == 0 || sum(q4) > 0) && rounds < n_rounds
        f_dense_time += @elapsed @finch begin
            q2 .= false
            for n2=_
                for n1=_
                    q2[n1] <<choose(false)>>= ((q1[n2] & A[n1, n2]) & !p1[n1])
                end
            end

            p2 .= false
            for n1=_
                p2[n1] = q2[n1] | p1[n1]
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
        p1 = Tensor(Dense(Element(false)), p4)
        q1 = Tensor(Dense(Element(false)), q4)
        rounds += 1
    end
    push!(f_dense_times, f_dense_time)
    global dense_p4 = p4
end

A_g = Materialize(t_undef, t_undef, :n1, :n2, Input(A, :n1, :n2))

println("Inserting Stats: $matrix")

println(@elapsed insert_statistics!(DCStats, A_g))

function get_bfs_query(A, q, p)
    q2 = Materialize(t_undef, :n1,
                        Aggregate(choose(false), :n2,
                            MapJoin(&,  Input(A, :n1, :n2),
                                        Input(q, :n2),
                                        MapJoin(!, Input(p, :n1)))))
    p2 = Materialize(t_undef, :n3, MapJoin(|,
                                        Input(q2, :n3),
                                        Input(p, :n3)))
    q3 = Materialize(t_undef, :n4,
                        Aggregate(choose(false), :n5,
                            MapJoin(&,  Input(A, :n4, :n5),
                                        Input(q2, :n5),
                                        MapJoin(!, Input(p2, :n4)))))
    p3 = Materialize(t_undef, :n6, MapJoin(|,
                                        Input(q3, :n6),
                                        Input(p2, :n6)))
    q4 = Materialize(t_undef, :n7,
                        Aggregate(choose(false), :n8,
                            MapJoin(&,  Input(A, :n7, :n8),
                                        Input(q3, :n8),
                                        MapJoin(!, Input(p3, :n7)))))
    p4 = Materialize(t_undef, :n9, MapJoin(|,
                                        Alias(:q4, :n9),
                                        Input(p3, :n9)))
    return [Query(:q4, q4), Query(:out, p4)]
end


println("Running Galley (non-CSE) Version: $matrix")
execute_times = []
opt_times = []
for i in 1:3
    rounds = 0
    global q_g = Materialize(t_undef, :n1, Input(q, :n1))
    global p_g = Materialize(t_undef, :n1, Input(p, :n1))
    opt_time = 0
    execute_time = 0
    verbose = (i == 1) ? 3 : 0
    while (rounds == 0 || sum(q) > 0) && rounds < n_rounds
        query = get_bfs_query(A_g, q_g, p_g)
        global result_galley = galley(query, simple_cse=false, faq_optimizer=greedy, ST=DCStats, max_kernel_size=4, verbose=verbose)
        execute_time += result_galley.execute_time
        opt_time += result_galley.opt_time
        global q_g = Materialize(t_undef, :n1, Input(result_galley.value[1], :n1))
        global p_g = Materialize(t_undef, :n1, Input(result_galley.value[2], :n1))
        rounds += 1
    end
    push!(opt_times, opt_time)
    push!(execute_times, execute_time)
end

println("Running Galley (CSE) Version: $matrix")
execute_times_cse = []
opt_times_cse = []
for i in 1:3
    rounds = 0
    global q_g = Materialize(t_undef, :n1, Input(q, :n1))
    global p_g = Materialize(t_undef, :n1, Input(p, :n1))
    opt_time = 0
    execute_time = 0
    verbose = i == 1 ? 3 : 0
    while (rounds == 0 || sum(q) > 0) && rounds < n_rounds
        query = get_bfs_query(A_g, q_g, p_g)
        global result_galley_cse = galley(query, simple_cse=true, faq_optimizer=greedy, ST=DCStats, max_kernel_size=4, verbose=verbose)
        execute_time += result_galley_cse.execute_time
        opt_time += result_galley_cse.opt_time
        global q_g = Materialize(t_undef, :n1, Input(result_galley_cse.value[1], :n1))
        global p_g = Materialize(t_undef, :n1, Input(result_galley_cse.value[2], :n1))
        rounds += 1
    end
    push!(opt_times_cse, opt_time)
    push!(execute_times_cse, execute_time)
end
println("Galley (Non-CSE) Exec: $(minimum(execute_times))")
println("Galley (Non-CSE) Opt: $(minimum(opt_times))")
println("Galley (CSE) Exec: $(minimum(execute_times_cse))")
println("Galley (CSE) Opt: $(minimum(opt_times_cse))")
println("Finch Sparse Exec: $(minimum(f_times))")
println("Finch Dense Exec: $(minimum(f_dense_times))")
println("F = G: $(sum(sparse_p4) == sum(result_galley.value[2]))")
println("F = G: $(sum(dense_p4) == sum(result_galley_cse.value[2]))")

#=
using DuckDB
dbconn = DBInterface.connect(DuckDB.DB, ":memory:")
result_duckdb = galley(deepcopy(Query(:out, g_p4)), dbconn=dbconn, ST=DCStats, faq_optimizer=pruned, verbose=3)
println("DuckDB == Galley:", result_duckdb.value == result_galley.value)
println("DuckDB Opt & Execute: ", result_duckdb.opt_time, "   ", result_duckdb.execute_time)
=#
