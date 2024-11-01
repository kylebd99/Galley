using Finch
using BenchmarkTools
using LinearAlgebra
include("../Experiments.jl")

function gen_mats(ni, nj, nk, sparsity)
    A = zeros(ni, nj)
    B = zeros(nk, nj)
    for i in 1:ni
        for j in 1:nj
            if rand() < (nj-j)/nj * sparsity
                A[i, j] = 1.0
            end
        end
    end
    for k in 1:nk
        for j in 1:nj
            if rand() < (nj-j)/nj * sparsity
                B[k, j] = 1.0
            end
        end
    end

    A = Tensor(Dense(SparseList(Element(0.0))), A)
    B = Tensor(Dense(SparseList(Element(0.0))), B)
    A = dropfills(A)
    B = dropfills(B)
    return A, B
end

function finch_spgemm(A, B; dense=false)
    times = []
    sparsity = 1.0
    for _ in 1:5
        C = dense ? Tensor(Dense(Dense(Element(0.0)))) : Tensor(Dense(Sparse(Element(0.0))))
        C_out = Tensor(Dense(SparseList(Element(0.0))))
        time = @elapsed @finch begin
            C .= 0
            for j=_
                for i=_
                    for k=_
                        C[k, i] += A[i,j]*B[k,j]
                    end
                end
            end
        end
        time += @elapsed dropfills!(C_out, C)
        sparsity = countstored(C) / prod(size(C))
        push!(times, time)
    end
    return mean(times[2:5]), sparsity
end

function galley_spgemm(A, B)
    C = Query(:C, Mat(:k, :i, Σ(:j, MapJoin(*, Input(A, :i, :j), Input(B, :k, :j)))))
    exec_times = []
    for iter in 1:5
        C_out = Tensor(Dense(SparseList(Element(0.0))))
        verbose = iter == 1 ? 3 : 0
        result = galley(C, verbose=verbose)
        time = result.execute_time
        time += @elapsed dropfills!(C_out, result.value)
        push!(exec_times, time)
    end
    return mean(exec_times[2:5])
end

function main()
    sparsities = [2.0^-(i) for i in reverse(4:8)]
    println(sparsities)
    galley_times = []
    dense_times = []
    sparse_times = []
    output_sparsity = []
    ni = 25000
    nj = 25000
    nk = 50
    for s in sparsities
        A, B = gen_mats(ni, nj, nk, s)
        println("------------------- Sparsity: $(countstored(A)/nj/ni) -------------------")
        push!(galley_times, galley_spgemm(A, B))
        push!(dense_times,  finch_spgemm(A, B, dense=true)[1])
        s_times, os = finch_spgemm(A, B, dense=false)
        push!(sparse_times, s_times)
        push!(output_sparsity, os)
        println("Output Sparsity: $(os)")
        println("Galley: $(galley_times[end])")
        println("Dense: $(dense_times[end])")
        println("Sparse: $(sparse_times[end])")
    end
    println(output_sparsity)
    plt = plot(output_sparsity, galley_times, lw=3, label="Galley")
    plot!(plt, output_sparsity, dense_times, lw=3, label="Dense")
    plot!(plt, output_sparsity, sparse_times, lw=3, label="Sparse")
    xaxis!(plt, xscale=:log10, xlabel="Output Sparsity")
    yaxis!(plt, yscale=:log10, ylabel="Execution Time")
    savefig(plt, "Experiments/Figures/spgemm.png")
end

main()
