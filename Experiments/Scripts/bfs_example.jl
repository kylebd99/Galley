include("../Experiments.jl")
using Finch
using Galley
using Galley: t_dense, t_undef, insert_statistics!
using MatrixDepot
using SparseArrays
using DelimitedFiles
using Statistics: mean


function run_exps(matrices)
    results = [("Method", "Dataset", "Runtime", "OptTime", "Iterations")]
    for matrix in matrices
        println("Running BFS for $matrix")
        println("Loading Data: $matrix")
        A = Tensor(Dense(SparseList(Element(false))), SparseMatrixCSC(matrixdepot(matrix)) .!= 0)
        n = size(A)[1]
        p = Tensor(SparseList(Element(false)), SparseVector(fsprand(Bool, n, 10)) .!= 0)
        q = Tensor(SparseList(Element(false)), p)

        n_reps = 5
        n_rounds = Inf
        println("Running Sparse Version: $matrix")
        sparse_p4 = nothing
        f_times = []
        for i in 1:n_reps
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
                            q2[n1] <<choose(false)>>= ((q1[n2] & A[walk(n1), n2]) & !p1[follow(n1)])
                        end
                    end

                    p2 .= false
                    for n1=_
                        p2[n1] = q2[n1] | p1[n1]
                    end

                    q3 .= false
                    for n2=_
                        for n1=_
                            q3[n1] <<choose(false)>>= ((q2[n2] & A[walk(n1), n2]) & !p2[follow(n1)])
                        end
                    end

                    p3 .= false
                    for n1=_
                        p3[n1] = q3[n1] | p2[n1]
                    end

                    q4 .= false
                    for n2=_
                        for n1=_
                            q4[n1] <<choose(false)>>= ((q3[n2] & A[walk(n1), n2]) & !p3[follow(n1)])
                        end
                    end

                    p4 .= false
                    for n1=_
                        p4[n1] = q4[n1] | p3[n1]
                    end
                end
                f_time += @elapsed p1 = Tensor(SparseList(Element(false)), p4)
                f_time += @elapsed q1 = Tensor(SparseList(Element(false)), q4)
                rounds += 1
            end
            push!(f_times, f_time)
            sparse_p4 = p4
        end

        println("Running Dense Version: $matrix")
        dense_p4 = nothing
        f_dense_times = []
        for i in  1:n_reps
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
                f_dense_time += @elapsed p1 = Tensor(Dense(Element(false)), p4)
                f_dense_time += @elapsed q1 = Tensor(Dense(Element(false)), q4)
                rounds += 1
            end
            push!(f_dense_times, f_dense_time)
            dense_p4 = p4
        end

        println("Running Hand Opt Version: $matrix")
        handopt_p4 = nothing
        f_handopt_times = []
        for i in  1:n_reps
            p1 = Tensor(Dense(Element(false)), p)
            q1 = Tensor(SparseByteMap(Element(false)), q)
            p2 = Tensor(Dense(Element(false)))
            q2 = Tensor(SparseByteMap(Element(false)))
            p3 = Tensor(Dense(Element(false)))
            q3 = Tensor(SparseByteMap(Element(false)))
            p4 = Tensor(Dense(Element(false)))
            q4 = Tensor(SparseByteMap(Element(false)))
            rounds = 0
            f_handopt_time = 0
            while  (rounds == 0 || sum(q4) > 0) && rounds < n_rounds
                f_handopt_time += @elapsed @finch begin
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
                f_handopt_time += @elapsed p1 = Tensor(Dense(Element(false)), p4)
                f_handopt_time += @elapsed q1 = Tensor(SparseByteMap(Element(false)), q4)
                rounds += 1
            end
            push!(f_handopt_times, f_handopt_time)
            handopt_p4 = p4
        end

        A_g = Materialize(t_undef, t_undef, :n1, :n2, Input(A, :n1, :n2))

        println("Inserting Stats: $matrix")

        println(@elapsed insert_statistics!(DCStats, A_g))

        function get_bfs_one_iter_query(A, q, p)
            q2 = Materialize(t_undef, :n1,
                                Aggregate(choose(false), :n2,
                                    MapJoin(&,  Input(A, :n1, :n2),
                                                Input(q, :n2),
                                                MapJoin(!, Input(p, :n1)))))
            p2 = Materialize(t_undef, :n3, MapJoin(|,
                                                Input(q2, :n3),
                                                Input(p, :n3)))
            return [Query(:q4, q2), Query(:out, p2)]
        end


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
        result_galley = nothing
        for i in 1:n_reps
            rounds = 0
            q_g = Materialize(t_undef, :n1, Input(q, :n1))
            p_g = Materialize(t_undef, :n1, Input(p, :n1))
            opt_time = 0
            execute_time = 0
            verbose = (i == 1) ? 0 : 0
            while (rounds == 0 || sum(q_g.expr.tns.val) > 0) && rounds < n_rounds
                query = get_bfs_query(A_g, q_g, p_g)
                result_galley = galley(query, simple_cse=false, faq_optimizer=pruned, ST=DCStats, verbose=verbose)
                execute_time += result_galley.execute_time
                opt_time += result_galley.opt_time
                q_g = Materialize(t_undef, :n1, Input(result_galley.value[1], :n1))
                p_g = Materialize(t_undef, :n1, Input(result_galley.value[2], :n1))
                rounds += 1
            end
            push!(opt_times, opt_time)
            push!(execute_times, execute_time)
        end

        println("Running Galley (one iter) Version: $matrix")
        execute_times_one_iter = []
        opt_times_one_iter = []
        result_galley_one_iter = nothing
        for i in 1:n_reps
            rounds = 0
            q_g = Materialize(t_undef, :n1, Input(q, :n1))
            p_g = Materialize(t_undef, :n1, Input(p, :n1))
            opt_time = 0
            execute_time = 0
            verbose = (i == 1) ? 0 : 0
            while (rounds == 0 || sum(q_g.expr.tns.val) > 0) && rounds < n_rounds * 3
                query = get_bfs_one_iter_query(A_g, q_g, p_g)
                result_galley_one_iter = galley(query, simple_cse=false, faq_optimizer=pruned, ST=DCStats, verbose=verbose)
                execute_time += result_galley_one_iter.execute_time
                opt_time += result_galley_one_iter.opt_time
                q_g = Materialize(t_undef, :n1, Input(result_galley_one_iter.value[1], :n1))
                p_g = Materialize(t_undef, :n1, Input(result_galley_one_iter.value[2], :n1))
                rounds += 1
            end
            push!(opt_times_one_iter, opt_time)
            push!(execute_times_one_iter, execute_time)
        end

        println("Running Galley (CSE) Version: $matrix")
        execute_times_cse = []
        opt_times_cse = []
        result_galley_cse = nothing
        for i in 1:n_reps
            rounds = 0
            q_g = Materialize(t_undef, :n1, Input(q, :n1))
            p_g = Materialize(t_undef, :n1, Input(p, :n1))
            opt_time = 0
            execute_time = 0
            verbose = (i == 1) ? 0 : 0
            while (rounds == 0 || sum(q_g.expr.tns.val) > 0) && rounds < n_rounds
                query = get_bfs_query(A_g, q_g, p_g)
                result_galley_cse = galley(query, simple_cse=true, faq_optimizer=pruned, ST=DCStats, verbose=verbose)
                execute_time += result_galley_cse.execute_time
                opt_time += result_galley_cse.opt_time
                q_g = Materialize(t_undef, :n1, Input(result_galley_cse.value[1], :n1))
                p_g = Materialize(t_undef, :n1, Input(result_galley_cse.value[2], :n1))
                rounds += 1
            end
            push!(opt_times_cse, opt_time)
            push!(execute_times_cse, execute_time)
        end
        println("Galley (Non-CSE) Exec: $(minimum(execute_times[2:end])), $(mean(execute_times[2:end])), $(maximum(execute_times[2:end]))")
        println("Galley (Non-CSE) Opt: $(mean(opt_times[2:end]))")
        println("Galley (One Iter) Exec: $(minimum(execute_times_one_iter[2:end])), $(mean(execute_times_one_iter[2:end])), $(maximum(execute_times_one_iter[2:end]))")
        println("Galley (One Iter) Opt: $(mean(opt_times_one_iter[2:end]))")
        println("Galley (CSE) Exec: $(minimum(execute_times_cse[2:end])), $(mean(execute_times_cse[2:end])), $(maximum(execute_times_cse[2:end]))")
        println("Galley (CSE) Opt: $(mean(opt_times_cse[2:end]))")
        println("Finch Sparse Exec: $(mean(f_times[2:end]))")
        #println("Finch SparseByteMap Exec: $(minimum(f_sparsebytemap_times))")
        println("Finch Dense Exec: $(mean(f_dense_times[2:end]))")
        println("Finch HandOpt Exec: $(mean(f_handopt_times[2:end]))")
        println("F = G: $(sum(sparse_p4) == sum(result_galley.value[2]))")
        #println("F = G: $(sum(sparsebytemap_p4) == sum(result_galley.value[2]))")
        println("F = G: $(sum(handopt_p4) == sum(result_galley.value[2]))")
        println("F = G: $(sum(dense_p4) == sum(result_galley_cse.value[2]))")
        push!(results, ("Sparse", matrix, string(mean(f_times[2:end])), string(0), string(n_rounds*3)))
        push!(results, ("Dense", matrix, string(mean(f_dense_times[2:end])), string(0), string(n_rounds*3)),)
        push!(results, ("HandOpt", matrix, string(mean(f_handopt_times[2:end])), string(0), string(n_rounds*3)))
        push!(results, ("Galley (No-CSE)", matrix, string(mean(execute_times[2:end])), string(mean(opt_times[2:end])), string(n_rounds*3)))
        push!(results, ("Galley (One Iter)", matrix, string(mean(execute_times_one_iter[2:end])), string(mean(opt_times_one_iter[2:end])), string(n_rounds*3)))
        push!(results, ("Galley (CSE)", matrix, string(mean(execute_times_cse[2:end])), string(mean(opt_times_cse[2:end])), string(n_rounds*3)))
    end
    writedlm("Experiments/Results/bfs.csv", results, ',')
end


matrix = "DIMACS10/kron_g500-logn16"
matrix = "SNAP/roadNet-CA"
matrix = "SNAP/com-LiveJournal"
matrix = "SNAP/soc-Epinions1"
matrix = "SNAP/com-Orkut"
matrices = ["DIMACS10/kron_g500-logn16",
            "SNAP/roadNet-CA",
            "SNAP/com-LiveJournal",
            "SNAP/soc-Epinions1",
            "SNAP/com-Orkut"]
run_exps(matrices)


using CSV
using DataFrames
using CategoricalArrays
using Measurements
using StatsPlots
using Plots
data = CSV.read("Experiments/Results/bfs.csv", DataFrame)
matrix_names = Dict("DIMACS10/kron_g500-logn16"=>"Kron", "SNAP/roadNet-CA"=>"RoadNet", "SNAP/com-LiveJournal"=>"LiveJournal", "SNAP/soc-Epinions1"=>"Epinions", "SNAP/com-Orkut"=>"Orkut")
#data = data[any.(zip(data.Method .== "Galley (One Iter)", data.Method .== "Sparse", data.Method .== "Dense", data.Method .== "HandOpt")) , :]
data.Dataset .= [matrix_names[x] for x in data.Dataset]
data.Method[data.Method .== "Galley (One Iter)"]  .= "Galley"
galley_opt_data = data[data.Method .== "Galley", :]
galley_opt_data.Method .= "Galley (Opt Time)"
galley_opt_data.Runtime .= galley_opt_data.OptTime
galley_exec_data = data[data.Method .== "Galley", :]
galley_exec_data.Method .= "Galley (Exec Time)"
#data[data.Method .== "Galley", :Runtime] .= data[data.Method .== "Galley", :].OptTime .+ data[data.Method .== "Galley", :].Runtime
data = data[any.(zip(data.Method .== "Galley", data.Method .== "Galley (Opt Time)", data.Method .== "Galley (Exec Time)", data.Method .== "Sparse", data.Method .== "Dense")) , :]
#append!(data, galley_opt_data)
#append!(data, galley_exec_data)
ordered_methods = CategoricalArray(data.Method)
levels!(ordered_methods, ["Galley", "Galley (Exec Time)", "Galley (Opt Time)", "Galley (No-CSE)", "Galley (CSE)", "Galley (One Iter)",  "Dense", "Sparse", "HandOpt"])
gbplot = StatsPlots.groupedbar(data.Dataset,
                                data.Runtime,
                                group = ordered_methods,
                                ylims=[10^-2, 10^2],
                                yscale=:log,
                                legend = :topleft,
                                size = (1000, 600),
                                ylabel = "Execute Time (s)",
                                xtickfontsize=16,
                                ytickfontsize=16,
                                xguidefontsize=18,
                                yguidefontsize=18,
                                legendfontsize=18,
                                xrotation = 25,
                                left_margin=10Plots.mm,
                                bottom_margin=15Plots.mm,
                                top_margin=10Plots.mm
                                )
savefig(gbplot, "Experiments/Figures/bfs.png")
