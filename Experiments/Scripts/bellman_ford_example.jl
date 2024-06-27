include("../Experiments.jl")
using Finch
using Galley
using Galley: t_dense, t_undef, insert_statistics!
using MatrixDepot
using SparseArrays
using DelimitedFiles


function run_exps(matrices)
    results = [("Method", "Dataset", "Runtime", "OptTime", "Iterations")]
    for matrix in matrices
        println("Running BFS for $matrix")
        println("Loading Data: $matrix")
        edges = redefault!(Tensor(SparseMatrixCSC(matrixdepot(matrix))), Inf)
        n_reps = 2
        n_sources = 1
        (n, m) = size(edges)
        @assert n == m
        sources = (rand(UInt, n_sources) .% n) .+ 1
        f_times = []
        f_times_sparse = []
        g_times = []
        g_opt_times = []
        for rep in 1:n_reps
            println("REP: $rep")
            dists_prev = Tensor(Dense(Dense(Element(Inf))), n, n_sources)
            dists = Tensor(Dense(Dense(Element(Inf))), n, n_sources)
            active_prev = Tensor(Dense(Dense(Element(Inf))), n, n_sources)
            active = Tensor(Dense(Dense(Element(Inf))), n, n_sources)
            for j in 1:n_sources
                dists_prev[sources[j], j] = 0
                active_prev[sources[j], j] = 1
            end
            f_time = 0
            for iter = 1:n
                f_time += @elapsed begin
                @finch (dists .= Inf; for k=_ for j=_; dists[j,k] <<min>>= dists_prev[j,k] end end)
                @finch begin
                    for k=_
                        for j = _
                            for i = _
                                dists[i,k] <<min>>= max(active_prev[j,k], dists_prev[follow(j),follow(k)] + edges[i, j])
                            end
                        end
                    end
                end
                @finch begin
                    active .= Inf
                    for k=_
                        for j = _
                            for i = _
                                active[i,k] <<min>>= max(active_prev[j,k], ifelse((dists_prev[follow(j),follow(k)] + edges[i, j]) < dists_prev[follow(i),follow(k)], 0, Inf))
                            end
                        end
                    end
                end
                active = dropfills(active)
                end

                if sum(active .< Inf) == 0
                    break
                end
                dists_prev, dists = dists, dists_prev
                active_prev, active = active, active_prev
            end
            push!(f_times, f_time)


            dists_prev = Tensor(Dense(Sparse(Element(Inf))), n, n_sources)
            dists = Tensor(Dense(Sparse(Element(Inf))), n, n_sources)
            active_prev = Tensor(Dense(Sparse(Element(Inf))), n, n_sources)
            active = Tensor(Dense(Sparse(Element(Inf))), n, n_sources)
            for j in 1:n_sources
                dists_prev[sources[j], j] = 0
                active_prev[sources[j], j] = 1
            end
            f_time_sparse = 0
            for iter = 1:n
                f_time_sparse += @elapsed begin
                @finch (dists .= Inf; for k=_ for j=_; dists[j,k] <<min>>= dists_prev[j,k] end end)
                @finch begin
                    for k=_
                        for j = _
                            for i = _
                                dists[i,k] <<min>>= max(active_prev[j,k], dists_prev[follow(j),follow(k)] + edges[i, j])
                            end
                        end
                    end
                end
                @finch begin
                    active .= Inf
                    for k=_
                        for j = _
                            for i = _
                                active[i,k] <<min>>= max(active_prev[j,k], ifelse((dists_prev[follow(j),follow(k)] + edges[i, j]) < dists_prev[follow(i),follow(k)], 0, Inf))
                            end
                        end
                    end
                end
                active = dropfills(active)
                end

                if sum(active .< Inf) == 0
                    break
                end
                dists_prev, dists = dists, dists_prev
                active_prev, active = active, active_prev
            end
            push!(f_times_sparse, f_time_sparse)



            g_dists_prev = Tensor(Dense(Dense(Element(Inf))), n, n_sources)
            g_active_prev = Tensor(Dense(SparseByteMap(Element(Inf))), n, n_sources)
            for j in 1:n_sources
                g_dists_prev[sources[j], j] = 0
                g_active_prev[sources[j], j] = 1
            end
            edges_g = Materialize(t_undef, t_undef, :i, :j, Input(edges, :i, :j))
            insert_statistics!(DCStats, edges_g)
            g_time = 0
            g_opt_time = 0
            for iter = 1:n
                g_dists2 = Materialize(t_undef, t_undef, :i, :k, MapJoin(min, Input(g_dists_prev, :i, :k), Aggregate(min, :j, MapJoin(max, Input(g_active_prev, :j, :k), MapJoin(+, Input(g_dists_prev, :j, :k), Input(edges_g, :i, :j))))))
                g_active2 = Materialize(t_undef, t_undef, :i, :k, Aggregate(min, :j, MapJoin(max, Input(g_active_prev, :j, :k), Input(edges_g, :i, :j), MapJoin(ifelse, MapJoin(<, Alias(:g_dists2, :i, :k) , Input(g_dists_prev, :i, :k)), 1, Inf))))
                results_galley = galley([Query(:g_dists2, g_dists2), Query(:g_active_2, g_active2)], verbose=0)
                g_time += results_galley.execute_time
                g_opt_time += results_galley.opt_time
                g_dists_prev = results_galley.value[1]
                g_active_prev = results_galley.value[2]
                if sum(pattern!(g_active_prev)) == 0
                    break
                end
            end
            push!(g_times, g_time)
            push!(g_opt_times, g_opt_time)

            println(g_dists_prev == dists)
        end
        println("Dense: $(minimum(f_times))")
        println("Sparse: $(minimum(f_times_sparse))")
        println("Galley: $(minimum(g_times))")
        println("Galley Opt: $(minimum(g_opt_times))")
    end
end


matrix = "DIMACS10/kron_g500-logn16"
matrix = "SNAP/roadNet-CA"
matrix = "SNAP/com-LiveJournal"
matrix = "SNAP/soc-Epinions1"
matrix = "SNAP/com-Orkut"
matrices = ["SNAP/roadNet-CA",
            "SNAP/com-LiveJournal",
            "SNAP/soc-Epinions1",
            "SNAP/com-Orkut"]
run_exps(matrices)

#=
using CSV
using DataFrames
data = CSV.read("Experiments/Results/bfs.csv", DataFrame)
gbplot = StatsPlots.groupedbar(data.Dataset,
                                data.Runtime,
                                group = data.Method,
                                ylims=[10^-4, 10^2],
                                yscale=:log,
                                legend = :outertopleft,
                                size = (1400, 600),
                                ylabel = "Runtime")
savefig(gbplot, "Experiments/Figures/bfs.png") =#
