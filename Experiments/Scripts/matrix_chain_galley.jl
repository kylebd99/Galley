using Galley
using Finch
using DataFrames
using CSV

function main()
    N = 2000
    densities = [.1, .01, .001, .0001, .00001, .000001]
    forward_times = []
    backward_times =[]
    sum_times = []
    elementwise_times = []
    for d in densities
        A = Tensor(Dense(SparseList(Element(0.0))), fsprand(N, N, .5))
        B = Tensor(Dense(SparseList(Element(0.0))), fsprand(N, N, .5))
        A_dense = Tensor(Dense(Dense(Element(0.0))), rand(N, N))
        B_dense = Tensor(Dense(Dense(Element(0.0))), rand(N, N))
        C = Tensor(Dense(SparseList(Element(0.0))), fsprand(N, N, d))
        n_reps = 7
        avg_time_forward = 0
        avg_time_backward = 0
        avg_time_sum = 0
        avg_time_elementwise = 0
        for i in 1:n_reps
            verbosity = i == 2 ? 3 : 0
            if i == 3
                avg_time_forward = 0
                avg_time_backward = 0
                avg_time_sum = 0
                avg_time_elementwise = 0
            end

            E1 = Query(:E1, Mat(:i, :l, Σ(:j, :k, MapJoin(*, Input(A, :i, :j), Input(B, :j, :k), Input(C, :k, :l)))))
            start_time = time()
            galley(E1, verbose=verbosity, faq_optimizer=greedy)
            end_time = time()
            avg_time_forward += end_time - start_time
            
            E2 = Query(:E2,  Mat(:i, :l, Σ(:j, :k, MapJoin(*, Input(C, :i, :j), Input(B, :j, :k), Input(A, :k, :l)))))
            start_time = time()
            galley(E2, verbose=verbosity, faq_optimizer=greedy)
            end_time = time()
            avg_time_backward += end_time - start_time

            E3 = Query(:E3,  Mat(Σ(:i, :j, :k, :l, MapJoin(*, Input(C, :i, :j), Input(B, :j, :k), Input(A, :k, :l)))))
            start_time = time()
            galley(E3, verbose=verbosity, faq_optimizer=greedy)
            end_time = time()
            avg_time_sum += end_time - start_time

            E4 = Query(:E4,  Mat(:i, :j, MapJoin(*, Input(A_dense, :i, :j), Input(B_dense, :i, :j), Input(C, :i, :j))))
            start_time = time()
            galley(E4, verbose=verbosity, faq_optimizer=greedy)
            end_time = time()
            avg_time_elementwise += end_time - start_time
        end
        avg_time_forward /= n_reps - 2
        avg_time_backward /= n_reps - 2
        avg_time_sum /= n_reps - 2
        avg_time_elementwise /= n_reps - 2
        push!(forward_times, avg_time_forward)
        push!(backward_times, avg_time_backward)
        push!(sum_times, avg_time_sum)
        push!(elementwise_times, avg_time_elementwise)
        println("Density: ", d)
        println("Forward Time: ", avg_time_forward)
        println("Backward Time: ", avg_time_backward)
        println("Sum Time: ", avg_time_sum)
        println("Elementwise Time: ", avg_time_elementwise)
    end
    println(forward_times)
    println(backward_times)
    println(sum_times)
    println(elementwise_times)
    data = []
    for i in eachindex(densities)
        push!(data, (Method="Galley", Algorithm="ABC", Sparsity=densities[i], Runtime=forward_times[i]))
        push!(data, (Method="Galley", Algorithm="CBA", Sparsity=densities[i], Runtime=backward_times[i]))
        push!(data, (Method="Galley", Algorithm="SUM(ABC)", Sparsity=densities[i], Runtime=sum_times[i]))
        push!(data, (Method="Galley", Algorithm="A*B*C", Sparsity=densities[i], Runtime=elementwise_times[i]))
    end
    df = 
    CSV.write("Experiments/Results/mat_exps_galley.csv", DataFrame(data))
end

main()


using Plots
using StatsPlots
using DataFrames
using CSV

data = DataFrame(CSV.File("Experiments/Results/mat_exps_galley.csv"))
append!(data, DataFrame(CSV.File("Experiments/Results/mat_exps_pytorch_serial.csv")), promote=true)
append!(data, DataFrame(CSV.File("Experiments/Results/mat_exps_pytorch_parallel.csv")), promote=true)

abc_plt = @df data[data.Algorithm .=="ABC", :] bar([i%6 + floor(i/6)/4 for i in 0:17], :Runtime, group=:Method,
                                        xticks=([.25, 1.25, 2.25, 3.25, 4.25, 5.25], [".1", ".01", ".001", ".0001", ".00001", ".000001"]),
                                        yscale=:log10, fillrange = 10^(-3), ylims=[10^(-3), 100],
                                        xflip=false, lw=2, xtickfontsize=12,  bar_width=.25,
                                        ytickfontsize=12, legendfontsize=12, legend=:topright,
                                        xlabel="Density of C", ylabel="Runtime", title="Matrix Chain Multiplication (ABC)")
savefig(abc_plt, "Experiments/Figures/mat_chain_abc.png")


cba_plt = @df data[data.Algorithm .=="CBA", :] bar([i%6 + floor(i/6)/4 for i in 0:17], :Runtime, group=:Method,
                                        xticks=([.25, 1.25, 2.25, 3.25, 4.25, 5.25], [".1", ".01", ".001", ".0001", ".00001", ".000001"]),
                                        yscale=:log10, fillrange = 10^(-3), ylims=[10^(-3), 100],
                                        xflip=false, lw=2, xtickfontsize=12,  bar_width=.25,
                                        ytickfontsize=12, legendfontsize=12, legend=:topright,
                                        xlabel="Density of C", ylabel="Runtime", title="Matrix Chain Multiplication (CBA)")
savefig(cba_plt, "Experiments/Figures/mat_chain_cba.png")

sum_plt = @df data[data.Algorithm .=="SUM(ABC)", :] bar([i%6 + floor(i/6)/4 for i in 0:17], :Runtime, group=:Method,
                                        xticks=([.25, 1.25, 2.25, 3.25, 4.25, 5.25], [".1", ".01", ".001", ".0001", ".00001", ".000001"]),
                                        yscale=:log10, fillrange = 10^(-3), ylims=[10^(-3), 100],
                                        xflip=false, lw=2, xtickfontsize=12,  bar_width=.25,
                                        ytickfontsize=12, legendfontsize=12, legend=:topright,
                                        xlabel="Density of C", ylabel="Runtime", title="Sum of Matrix Chain (SUM(ABC))")
savefig(sum_plt, "Experiments/Figures/mat_chain_sum.png")

sum_plt = @df data[data.Algorithm .=="A*B*C", :] bar([i%6 + floor(i/6)/4 for i in 0:17], :Runtime, group=:Method,
                                        xticks=([.25, 1.25, 2.25, 3.25, 4.25, 5.25], [".1", ".01", ".001", ".0001", ".00001", ".000001"]),
                                        yscale=:log10, fillrange = 10^(-3), ylims=[10^(-3), 10],
                                        xflip=false, lw=2, xtickfontsize=12,  bar_width=.25,
                                        ytickfontsize=12, legendfontsize=12, legend=:topright,
                                        xlabel="Density of C", ylabel="Runtime", title="Elementwise Matrix Multiplication (A*B*C)")
savefig(sum_plt, "Experiments/Figures/mat_chain_elementwise.png")
