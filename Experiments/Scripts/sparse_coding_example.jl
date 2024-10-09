using Finch
using Galley: insert_statistics!, t_undef
using Galley
using Statistics
include("../Experiments.jl")

# TODO: Replace this artificial data with real data from TPC-H or IMDB
nx = 300
ns = 100
nb = 100000
nc = 10000
np = 1000
S = Tensor(Dense(Sparse(Element(0.0))), fsprand(Int, nx, ns, 100*ns) .% 100)
BS = Tensor(Dense(Sparse(Element(false))), fsprand(Bool, nb, ns, nb))
X_s = Materialize(t_undef, t_undef, :i, :x, Σ(:s, MapJoin(*, Input(BS, :i,:s), Input(S, :x, :s))))
C = Tensor(Dense(Sparse(Element(0.0))), fsprand(Int, nx, nc, 100*nc) .% 100)
BC = Tensor(Dense(Sparse(Element(false))), fsprand(Bool, nb, nc, nb))
X_c = Materialize(t_undef, t_undef, :i, :x, Σ(:c, MapJoin(*, Input(BC, :i,:c), Input(C, :x, :c))))
P = Tensor(Dense(Sparse(Element(0.0))), fsprand(Int, nx, np, 100*np) .% 100)
BP = Tensor(Dense(Sparse(Element(false))), fsprand(Bool, nb, np, nb))
X_p = Materialize(t_undef, t_undef, :i, :x, Σ(:p, MapJoin(*, Input(BP, :i,:p), Input(P, :x, :p))))

# Here the feature vector is the results of a "database query" which combines information
# about stores (S), purchases (B), Customers (C), and products (P). Each row encodes the
# combined information about one purchase.
X_g = Materialize(t_undef, t_undef, :i, :x, MapJoin(+, X_s[:i, :x], X_c[:i, :x], X_p[:i, :x]))


function finch_sp_als(BS, S, BC, C, BP, P, u, v)
    X = Tensor(Dense(Sparse(Element(0.0))))
    l = Scalar(0.0)
    f_time = @elapsed @finch begin
        X .= 0
        for s=_
            for i =_
                for x =_
                    X[x, i] += BS[i, s] * S[x, s]
                end
            end
        end

        for c=_
            for i =_
                for x =_
                    X[x, i] += BC[i, c] * C[x, c]
                end
            end
        end

        for p=_
            for i =_
                for x =_
                    X[x, i] += BP[i, p] * P[x, p]
                end
            end
        end


        l.=0
        for i =_
            for x =_
                l[] += (X[x,i] - u[i]*v[x])^2
            end
        end
    end
    return f_time, l[]
end

X = Tensor(Dense(SparseList(Element(0.0))))
u = Tensor(Dense(Element(0.0)), rand(Float64, nb) .% 100)
v = Tensor(Dense(Element(0.0)), rand(Float64, nx) .% 100)
f_l = nothing
f_times = []
f_ls = []
for i in 1:5
    f_time, l = finch_sp_als(BS, S, BC, C, BP, P, u, v)
    push!(f_times, f_time)
    push!(f_ls, l)
end

q = Materialize(Σ(:i, :j, MapJoin(*, MapJoin(+, Input(X_g, :i, :j), MapJoin(*, MapJoin(-, Input(u, :i)), Input(v, :j))),
                                     MapJoin(+, Input(X_g, :i, :j), MapJoin(*, MapJoin(-, Input(u, :i)), Input(v, :j))))))
insert_statistics!(DCStats, q)
g_times = []
g_opt_times = []
for i in 1:5
    result_galley = galley([Query(:out, q)], ST=DCStats,  verbose=0)
    push!(g_times, result_galley.execute_time)
    push!(g_opt_times, result_galley.opt_time)
end
result = galley([Query(:out, q)], ST=DCStats, verbose=3)
println(result)
println("Galley Exec: $(minimum(g_times))")
println("Galley Opt: $(minimum(g_opt_times))")
println("Finch Exec: $(minimum(f_times))")
println("F = G: $(mean(f_ls) - result.value[1])")


function finch_sp_lr(BS, S, BC, C, BP, P, Y, θ)
    X = Tensor(Dense(Sparse(Element(0.0))))
    sigma = Tensor(Dense(Dense(Element(0.0))))
    C2 = Tensor(Dense(Element(0.0)))
    st = Tensor(Dense(Element(0.0)))
    d = Tensor(Dense(Element(0.0)))
    f_time = @elapsed @finch begin
        X .= 0
        for s=_
            for i =_
                for x =_
                    X[x, i] += BS[i, s] * S[x, s]
                end
            end
        end

        for c=_
            for i =_
                for x =_
                    X[x, i] += BC[i, c] * C[x, c]
                end
            end
        end

        for p=_
            for i =_
                for x =_
                    X[x, i] += BP[i, p] * P[x, p]
                end
            end
        end

        sigma .= 0
        for i=_
            for k=_
                for j=_
                    sigma[j,k] += 1/nb * X[j, i] * X[k, i]
                end
            end
        end

        C2 .= 0
        for i=_
            for j=_
                C2[j] += X[j, i] * Y[i] * 1.0/nb
            end
        end


        st.= 0
        for k =_
            for j =_
                st[k] += sigma[j, k] * θ[k]
            end
        end

        d .= 0
        for k =_
            d[k] = st[k] + .1 * θ[k] - C2[k]
        end
    end
    return f_time, d
end

Y = Tensor(Dense(Element(0.0)), rand(Int, nb) .% 100)
θ = Tensor(Dense(Element(0.0)), rand(Int, nx) .% 100)
f_d = nothing
f_times = []
f_ls = []
for i in 1:5
    f_time, d = finch_sp_lr(BS, S, BC, C, BP, P, Y, θ)
    push!(f_times, f_time)
    push!(f_ls, d)
end

sigma = Materialize(t_undef, t_undef, :j, :k, MapJoin(*, 1/nb, Σ(:i, MapJoin(*, X_g[:i, :j],  X_g[:i, :k]))))
c = Materialize(t_undef, :j, MapJoin(*, -1/nb, Σ(:i, MapJoin(*, Input(Y, :i),  X_g[:i, :j]))))
sy = Materialize(MapJoin(*, 1/nb, Σ(:i, MapJoin(*, Input(Y, :i), Input(Y, :i)))))
d_g = Materialize(t_undef, :k, MapJoin(+, Σ(:j, MapJoin(*, sigma[:j, :k], Input(θ, :k))), MapJoin(*, .1, Input(θ, :k)), c[:k]))
#insert_statistics!(DCStats, d_g)
g_times = []
g_opt_times = []
for i in 1:5
    result_galley = galley([Query(:out, d_g)], ST=DCStats,  verbose=0)
    push!(g_times, result_galley.execute_time)
    push!(g_opt_times, result_galley.opt_time)
end
result = galley([Query(:out, d_g)], ST=DCStats, verbose=3)
#println(result)
println("Galley Exec: $(minimum(g_times))")
println("Galley Opt: $(minimum(g_opt_times))")
println("Finch Exec: $(minimum(f_times))")
println("F = G: $(sum(abs.(f_ls[1] - result.value[1])))")
