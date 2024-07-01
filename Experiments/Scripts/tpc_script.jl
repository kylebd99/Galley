using Finch
using Galley: insert_statistics!, t_undef
using Galley
using DataFrames
using CSV
using SparseArrays
using Statistics
include("../Experiments.jl")

# First, we load the TPCH data
customer = CSV.read("Experiments/Data/TPCH/customer.tbl", DataFrame, delim='|', header=[:CustomerKey, :Name, :Address, :NationKey, :Phone, :AcctBal, :MktSegment, :Comment, :Col9])
lineitem = CSV.read("Experiments/Data/TPCH/lineitem.tbl", DataFrame, delim='|', header=[:OrderKey, :PartKey, :SuppKey, :LineNumber, :Quantity, :ExtendedPrice, :Discount, :Tax, :ReturnFlag, :LineStatus, :ShipDate, :CommitDate, :ReceiptDate, :ShipInstruct, :ShipMode, :Comment])
orders = CSV.read("Experiments/Data/TPCH/orders.tbl", DataFrame, delim='|', header=[:OrderKey, :CustomerKey, :OrderStatus, :TotalPrice, :OrderDate, :OrderPriority, :Clerk, :ShipPriority, :Comment])
partsupp = CSV.read("Experiments/Data/TPCH/partsupp.tbl", DataFrame, delim='|', header=[:PartKey, :SuppKey, :AvailQty, :SupplyCost, :Comment])
part = CSV.read("Experiments/Data/TPCH/part.tbl", DataFrame, delim='|', header=[:PartKey, :Name, :MFGR, :Brand, :Type, :Size, :Container, :RetailPrice, :Comment])
supplier = CSV.read("Experiments/Data/TPCH/supplier.tbl", DataFrame, delim='|', header=[:SuppKey, :Name, :Address, :NationKey, :Phone, :AcctBal, :Comment])
nation = CSV.read("Experiments/Data/TPCH/nation.tbl", DataFrame, delim='|', header=[:NationKey, :Name, :RegionKey, :Comment])
region = CSV.read("Experiments/Data/TPCH/region.tbl", DataFrame, delim='|', header=[:RegionKey, :Name, :Comment])

orderkey_idx = Dict(o => i for (i, o) in enumerate(unique(orders.OrderKey)))
partkey_idx = Dict(p => i for (i, p) in enumerate(unique(part.PartKey)))
suppkey_idx = Dict(s => i for (i, s) in enumerate(unique(supplier.SuppKey)))
customerkey_idx = Dict(c => i for (i, c) in enumerate(unique(customer.CustomerKey)))

function simplify_col(df, x, val_to_idx)
    df[!, x] = [val_to_idx[x] for x in df[!, x]]
end

simplify_col(lineitem, :OrderKey, orderkey_idx)
simplify_col(lineitem, :PartKey, partkey_idx)
simplify_col(lineitem, :SuppKey, suppkey_idx)
simplify_col(orders, :OrderKey, orderkey_idx)
simplify_col(orders, :CustomerKey, customerkey_idx)
simplify_col(customer, :CustomerKey, customerkey_idx)
simplify_col(part, :PartKey, partkey_idx)
simplify_col(supplier, :SuppKey, suppkey_idx)



function one_hot_encode_col!(df, x)
    ux = unique(df[!, x])
    transform!(df, @. x => ByRow(isequal(ux)) .=> Symbol(string(x) * '_' * string(ux)))
    select!(df, Not(x))
end

function one_hot_encode_cols!(df, xs)
    for x in xs
        one_hot_encode_col!(df, x)
    end
end

function col_to_join_matrix(df, x)
    return sparse(1:nrow(df), df[!, x], ones(nrow(df)))
end

function align_x_dims(X, x_start, x_dim)
    I, J, V = Int[], Int[], Float64[]
    for i in 1:size(X)[1]
        for j in 1:size(X)[2]
            push!(I, i + x_start)
            push!(J, j)
            push!(V, X[i,j])
        end
    end
    new_X = SparseMatrixCSC(sparse(I, J, V, x_dim, size(X)[2]))
    return Tensor(Dense(SparseList(Element(0.0))), new_X)
end

# Conceptually, each row of X is going to be a line item.
lineitem = lineitem[!, [:OrderKey, :PartKey, :SuppKey, :LineNumber, :Quantity, :ExtendedPrice, :Discount, :Tax, :ReturnFlag, :LineStatus, :ShipMode]]
one_hot_encode_cols!(lineitem, [:ReturnFlag, :LineStatus, :ShipMode])
orders = orders[!, [:OrderKey, :CustomerKey, :OrderStatus, :TotalPrice, :OrderPriority, :ShipPriority]]
orders.CustomerKey = orders.CustomerKey .+ 1
one_hot_encode_cols!(orders, [:OrderStatus, :OrderPriority, :ShipPriority])
customer = customer[!, [:CustomerKey, :NationKey, :AcctBal, :MktSegment]]
one_hot_encode_cols!(customer, [:NationKey, :MktSegment])
partsupp = partsupp[!, [:PartKey, :SuppKey]]
supplier = supplier[!, [:SuppKey, :NationKey, :AcctBal]]
one_hot_encode_cols!(supplier, [:NationKey])
part = part[!, [:PartKey, :MFGR, :Brand, :Size, :Container, :RetailPrice]]
one_hot_encode_cols!(part, [:MFGR, :Brand, :Container])
lineitem_x = Matrix(select(lineitem, Not([:OrderKey, :PartKey, :SuppKey])))'
lineitem_y = Vector(lineitem[!, :ExtendedPrice])
li_order = Tensor(Dense(SparseList(Element(false))), col_to_join_matrix(lineitem, :OrderKey))
orders_x = Matrix(select(orders, Not([:OrderKey, :CustomerKey])))'
order_cust = Tensor(Dense(SparseList(Element(false))), col_to_join_matrix(orders, :CustomerKey))
customer_x = Matrix(select(customer, Not([:CustomerKey])))'
li_supp = Tensor(Dense(SparseList(Element(false))), col_to_join_matrix(lineitem, :SuppKey))
supplier_x = Matrix(select(supplier, Not([:SuppKey])))'
li_part = Tensor(Dense(SparseList(Element(false))), col_to_join_matrix(lineitem, :PartKey))
part_x = Matrix(select(part, Not([:PartKey])))'

x_starts = cumsum([0, size(lineitem_x)[1], size(orders_x)[1], size(customer_x)[1], size(supplier_x)[1], size(part_x)[1]])
x_dim = x_starts[end]
lineitem_x = align_x_dims(lineitem_x, x_starts[1], x_dim)
orders_x = align_x_dims(orders_x, x_starts[2], x_dim)
customer_x = align_x_dims(customer_x, x_starts[3], x_dim)
supplier_x = align_x_dims(supplier_x, x_starts[4], x_dim)
part_x = align_x_dims(part_x, x_starts[5], x_dim)

X_g = Materialize(t_undef, t_undef, :i, :j, MapJoin(+,  Input(lineitem_x, :j, :i, "lineitem_x"),
                                                        Σ(:o, MapJoin(*, Input(li_order, :i, :o, "li_order"), Input(orders_x, :j, :o, "orders_x"))),
                                                        Σ(:o, :c, MapJoin(*, Input(li_order, :i, :o, "li_order"), Input(order_cust, :o, :c, "order_cust"), Input(customer_x, :j, :c, "customer_x"))),
                                                        Σ(:s, MapJoin(*, Input(li_supp, :i, :s, "li_supp"), Input(supplier_x, :j, :s, "supplier_x"))),
                                                        Σ(:p, MapJoin(*, Input(li_part, :i, :p, "li_part"), Input(part_x, :j, :p, "part_x"))),))

nb = size(lineitem_x)[2]
Y = Tensor(Dense(Element(0.0)), lineitem_y)
θ = Tensor(Dense(Element(0.0)), ones(Int, size(lineitem_x)[1]) .% 100)
sigma = Materialize(t_undef, t_undef, :j, :k, MapJoin(*, 1/nb, Σ(:i, MapJoin(*, X_g[:i, :j],  X_g[:i, :k]))))
c = Materialize(t_undef, :j, MapJoin(*, -1/nb, Σ(:i, MapJoin(*, Input(Y, :i),  X_g[:i, :j]))))
sy = Materialize(MapJoin(*, 1/nb, Σ(:i, MapJoin(*, Input(Y, :i), Input(Y, :i)))))
d_g = Materialize(t_undef, :j, MapJoin(+, Σ(:k, MapJoin(*, sigma[:j, :k], Input(θ, :k))), MapJoin(*, .1, Input(θ, :j)), c[:j]))
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


function finch_sp_lr(lineitem_x, li_order, orders_x, order_cust, customer_x, li_supp, supplier_x, li_part, part_x, Y, θ)
    X = Tensor(Dense(Dense(Element(0.0))))
    sigma = Tensor(Dense(Dense(Element(0.0))))
    C2 = Tensor(Dense(Element(0.0)))
    st = Tensor(Dense(Element(0.0)))
    d = Tensor(Dense(Element(0.0)))
    f_time = @elapsed @finch begin
        X .= 0
        for i =_
            for x =_
                X[x, i] += lineitem_x[x, i]
            end
        end

        for o=_
            for i =_
                for x =_
                    X[x, i] += li_order[i, o] * orders_x[x, o]
                end
            end
        end

        for c=_
            for o=_
                for i =_
                    for x =_
                        X[x, i] += li_order[i, o] * order_cust[o, c] * customer_x[x, c]
                    end
                end
            end
        end

        for s=_
            for i =_
                for x =_
                    X[x, i] += li_supp[i, s] * supplier_x[x, s]
                end
            end
        end

        for p=_
            for i =_
                for x =_
                    X[x, i] += li_part[i, p] * part_x[x, p]
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
                st[j] += sigma[j, k] * θ[k]
            end
        end

        d .= 0
        for k =_
            d[k] = st[k] + .1 * θ[k] - C2[k]
        end
    end
    return f_time, d
end

f_d = nothing
f_times = []
f_ls = []
for i in 1:5
    f_time, d = finch_sp_lr(lineitem_x, li_order, orders_x, order_cust, customer_x, li_supp, supplier_x, li_part, part_x, lineitem_y, θ)
    push!(f_times, f_time)
    push!(f_ls, d)
end

println("Galley Exec: $(minimum(g_times))")
println("Galley Opt: $(minimum(g_opt_times))")
println("Finch Exec: $(minimum(f_times))")
println("∑|F_i - G_i|: $(sum(abs.(f_ls[1] - result.value[1])))")

function finch_sp_lr2(li_supp, supplier_x, li_part, part_x, li_y, θ)
    Y = Tensor(Dense(Sparse(Element(0.0))))
    X = Tensor(Dense(Sparse(Sparse(Element(0.0)))))
    li_supp_x = Tensor(Dense(Sparse(Element(0.0))))
    sigma = Tensor(Dense(Dense(Element(0.0))))
    C2 = Tensor(Dense(Element(0.0)))
    st = Tensor(Dense(Element(0.0)))
    d = Tensor(Dense(Element(0.0)))
    f_time = @elapsed @finch begin
        X .= 0
        Y .=0
        for p = _
            for i1 =_
                for i2 =_
                    for x =_
                        X[x, i2, i1] += li_part[i1, p] * li_part[i2, p] * part_x[x, p]
                    end
                    Y[i2, i1] += li_part[i1, p] * li_part[i2, p] * (li_y[i1] - li_y[i2])
                end
            end
        end
    end
    println("Initialized X and Y")
    f_time += @elapsed @finch begin
        li_supp_x .= 0
        for s=_
            for i1 =_
                for x =_
                    li_supp_x[x, i1] += li_supp[i1, s] * supplier_x[x, s]
                end
            end
        end
    end
    println("Made li_supp_x")
    f_time += @elapsed @finch begin
        for i1 =_
            for i2 =_
                for x =_
                    X[x, i2, i1] += Y[i2, i1] * li_supp_x[x, i1]
                end
            end
        end
    end
    println("Added Supp 1")
    f_time += @elapsed @finch begin
        for i1 =_
            for i2 =_
                for x =_
                    X[x, i2, i1] += Y[i2, i1] * li_supp_x[x, i2]
                end
            end
        end
    end
    println("Added Supp 2")
    f_time += @elapsed @finch begin
        sigma .= 0
        for i1=_
            for i2=_
                for k=_
                    for j=_
                        sigma[j,k] += 1/nb * X[j, i2, i1] * X[k, i2, i1]
                    end
                end
            end
        end
    end
    println("Computed Sigma")
    f_time += @elapsed @finch begin
        C2 .= 0
        for i1=_
            for i2=_
                for j=_
                    C2[j] += X[j, i2, i1] * Y[i2, i1] * 1.0/nb
                end
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

# This formulation makes each row of the output a pair of line items for the same part
# and includes information about their suppliers
X_g = Materialize(t_undef, t_undef, t_undef, :i1, :i2, :j, Σ(:p, MapJoin(*, Input(li_part, :i1, :p, "li_part"), Input(li_part, :i2, :p, "li_part"),
                                                        MapJoin(+,
                                                        Input(part_x, :j, :p, "part_x"),
                                                        Σ(:s1, MapJoin(*, Input(li_supp, :i1, :s1, "li_supp"), Input(supplier_x, :j, :s1, "supplier_x"))),
                                                        Σ(:s2, MapJoin(*, Input(li_supp, :i2, :s2, "li_supp"), Input(supplier_x, :j, :s2, "supplier_x")))))))

nb = size(lineitem_x)[2]
lineitem_y_f = Tensor(Dense(Element(0.0)), lineitem_y)
Y = Materialize(t_undef, t_undef, :i1, :i2, Σ(:p, MapJoin(*, Input(li_part, :i1, :p, "li_part"), Input(li_part, :i2, :p, "li_part"), MapJoin(+, Input(lineitem_y_f, :i1), MapJoin(-, Input(lineitem_y_f, :i2))))))
θ = Tensor(Dense(Element(0.0)), ones(Int, size(lineitem_x)[1]) .% 100)
sigma = Materialize(t_undef, t_undef, :j, :k, MapJoin(*, 1/nb, Σ(:i1, :i2, MapJoin(*, X_g[:i1, :i2, :j],  X_g[:i1, :i2, :k]))))
c = Materialize(t_undef, :j, MapJoin(*, -1/nb, Σ(:i1, :i2, MapJoin(*, Y[:i1, :i2],  X_g[:i1, :i2, :j]))))
sy = Materialize(MapJoin(*, 1/nb, Σ(:i1, :i2, MapJoin(*, Y[:i1, :i2], Input(Y, :i1, :i2)))))
d_g = Materialize(t_undef, :k, MapJoin(+, Σ(:j, MapJoin(*, sigma[:j, :k], Input(θ, :k, "θ"))), MapJoin(*, .1, Input(θ, :k, "θ")), c[:k]))
#insert_statistics!(DCStats, d_g)
g_times = []
g_opt_times = []
for i in 1:3
    result_galley = galley([Query(:out, d_g)], ST=DCStats,  verbose=0)
    push!(g_times, result_galley.execute_time)
    push!(g_opt_times, result_galley.opt_time)
end
result = galley([Query(:out, d_g)], ST=DCStats, verbose=3)

f_d = nothing
f_times = []
f_ls = []
for i in 1:3
    f_time, d = finch_sp_lr2(li_supp, supplier_x, li_part, part_x, lineitem_y, θ)
    push!(f_times, f_time)
    push!(f_ls, d)
    GC.gc()
end
println("Galley Exec: $(minimum(g_times))")
println("Galley Opt: $(minimum(g_opt_times))")
println("Finch Exec: $(minimum(f_times))")
println("F = G: $(sum(abs.(f_ls[1] - result.value[1])))")
