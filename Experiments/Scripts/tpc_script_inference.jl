using Finch
using Galley: insert_statistics!, t_undef
using Galley
using DataFrames
using CSV
using SparseArrays
using Statistics
using StatsFuns
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

function col_to_join_matrix(df, x, n_rows, n_cols)
    return sparse(1:nrow(df), df[!, x], ones(nrow(df)), n_rows, n_cols)
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
li_order = Tensor(Dense(SparseList(Element(false))), col_to_join_matrix(lineitem, :OrderKey, nrow(lineitem), maximum(values(orderkey_idx))))
#li_order = Tensor(Dense(SparseList(Element(false))), fsprand(Bool, size(li_order)..., 100000))
orders_x = Matrix(select(orders, Not([:OrderKey, :CustomerKey])))'
order_cust = Tensor(Dense(SparseList(Element(false))), col_to_join_matrix(orders, :CustomerKey, nrow(orders), maximum(values(customerkey_idx))))
#order_cust = Tensor(Dense(SparseList(Element(false))), fsprand(Bool, size(order_cust)..., 100000))
customer_x = Matrix(select(customer, Not([:CustomerKey])))'
li_supp = Tensor(Dense(SparseList(Element(false))), col_to_join_matrix(lineitem, :SuppKey, nrow(lineitem), maximum(values(suppkey_idx))))
#li_supp = Tensor(Dense(SparseList(Element(false))), fsprand(Bool, size(li_supp)..., 100000))
supplier_x = Matrix(select(supplier, Not([:SuppKey])))'
li_part = Tensor(Dense(SparseList(Element(false))), col_to_join_matrix(lineitem, :PartKey, nrow(lineitem), maximum(values(partkey_idx))))
#li_part = Tensor(Dense(SparseList(Element(false))), fsprand(Bool, size(li_part)..., 100000))
part_x = Matrix(select(part, Not([:PartKey])))'

x_starts = cumsum([0, size(orders_x)[1], size(customer_x)[1], size(supplier_x)[1], size(part_x)[1]])
x_dim = x_starts[end]
orders_x = align_x_dims(orders_x, x_starts[1], x_dim)
customer_x = align_x_dims(customer_x, x_starts[2], x_dim)
supplier_x = align_x_dims(supplier_x, x_starts[3], x_dim)
part_x = align_x_dims(part_x, x_starts[4], x_dim)

X_g = Materialize(t_undef, t_undef, :i, :j, MapJoin(+,  #Input(lineitem_x, :j, :i, "lineitem_x"),
                                                        Σ(:o, MapJoin(*, Input(li_order, :i, :o, "li_order"), Input(orders_x, :j, :o, "orders_x"))),
                                                        Σ(:o, :c, MapJoin(*, Input(li_order, :i, :o, "li_order"), Input(order_cust, :o, :c, "order_cust"), Input(customer_x, :j, :c, "customer_x"))),
                                                        Σ(:s, MapJoin(*, Input(li_supp, :i, :s, "li_supp"), Input(supplier_x, :j, :s, "supplier_x"))),
                                                        Σ(:p, MapJoin(*, Input(li_part, :i, :p, "li_part"), Input(part_x, :j, :p, "part_x"))),))

nb = size(lineitem_y)[1]
θ = Tensor(Dense(Element(0.0)), ones(Int, size(supplier_x)[1]) .% 100)
P = Materialize(t_undef, :i, Aggregate(+, :j, MapJoin(*, X_g[:i, :j], Input(θ, :j))))
#insert_statistics!(DCStats, d_g)
g_times = []
g_opt_times = []
result = galley([Query(:out, P)], ST=DCStats, verbose=3)
for i in 1:2
    result_galley = galley([Query(:out, P)], ST=DCStats,  verbose=0)
    push!(g_times, result_galley.execute_time)
    push!(g_opt_times, result_galley.opt_time)
end


function finch_sp_lr(li_order, orders_x, order_cust, customer_x, li_supp, supplier_x, li_part, part_x, θ)
    X = Tensor(Dense(Sparse(Element(0.0))))
    P = Tensor(Dense((Element(0.0))))
    f_time = @elapsed @finch begin
        X .= 0

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

        P .= 0
        for i=_
            for j=_
                P[i] +=  X[j, i] * θ[j]
            end
        end
    end
    return f_time, P
end

function finch_lr(li_order, orders_x, order_cust, customer_x, li_supp, supplier_x, li_part, part_x, θ)
    X = Tensor(Dense(Dense(Element(0.0))))
    P = Tensor(Dense((Element(0.0))))
    f_time = @elapsed @finch begin
        X .= 0

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

        P .= 0
        for i=_
            for j=_
                P[i] +=  X[j, i] * θ[j]
            end
        end
    end
    return f_time, P
end

f_d = nothing
f_times = []
f_ls = []
for i in 1:2
    f_time, p = finch_lr(li_order, orders_x, order_cust, customer_x, li_supp, supplier_x, li_part, part_x, θ)
    push!(f_times, f_time)
    push!(f_ls, p)
end

f_sp_times = []
f_sp_ls = []
for i in 1:2
    f_time, p = finch_sp_lr(li_order, orders_x, order_cust, customer_x, li_supp, supplier_x, li_part, part_x, θ)
    push!(f_sp_times, f_time)
    push!(f_sp_ls, p)
end

sigmoid(x) = 1.0 - exp(x-log1pexp(x))
p_query = Query(:out, Materialize(t_undef, :i, MapJoin(sigmoid, Σ(:k, MapJoin(*, X_g[:i, :k], Input(θ, :k))))))
g_log_times = []
g_log_opt_times = []
for i in 1:2
    result_galley = galley(p_query, ST=DCStats,  verbose=0)
    push!(g_log_times, result_galley.execute_time)
    push!(g_log_opt_times, result_galley.opt_time)
end
result_log = galley([p_query], ST=DCStats, verbose=3)

function finch_sp_log(li_order, orders_x, order_cust, customer_x, li_supp, supplier_x, li_part, part_x, θ)
    X = Tensor(Dense(Sparse(Element(0.0))))
    p_inter = Tensor(Dense(Element(0.0)))
    prediction = Tensor(Dense(Element(0.5)))
    f_time = @elapsed @finch begin
        X .= 0
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

        p_inter .= 0
        for i=_
            for k=_
                p_inter[i] += X[k, i] * θ[k]
            end
        end

        prediction .= 0.5
        for i = _
            prediction[i] = sigmoid(p_inter[i])
        end
    end
    return f_time, prediction
end


function finch_log(li_order, orders_x, order_cust, customer_x, li_supp, supplier_x, li_part, part_x, θ)
    X = Tensor(Dense(Dense(Element(0.0))))
    p_inter = Tensor(Dense(Element(0.0)))
    prediction = Tensor(Dense(Element(0.0)))
    f_time = @elapsed @finch begin
        X .= 0
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

        p_inter .= 0
        for i=_
            for k=_
                p_inter[i] += X[k, i] * θ[k]
            end
        end

        prediction .= 0
        for i = _
            prediction[i] = sigmoid(p_inter[i])
        end
    end
    return f_time, prediction
end


f_log_times = []
f_logs = []
for i in 1:2
    f_time, prediction = finch_log(li_order, orders_x, order_cust, customer_x, li_supp, supplier_x, li_part, part_x, θ)
    push!(f_log_times, f_time)
    push!(f_logs, prediction)
end

f_sp_log_times = []
f_sp_logs = []
for i in 1:2
    f_time, prediction = finch_sp_log(li_order, orders_x, order_cust, customer_x, li_supp, supplier_x, li_part, part_x, θ)
    push!(f_sp_log_times, f_time)
    push!(f_sp_logs, prediction)
end

println("Galley Exec: $(minimum(g_times))")
println("Galley Opt: $(minimum(g_opt_times))")
println("Finch (Dense) Exec: $(minimum(f_times))")
println("Finch (Sparse) Exec: $(minimum(f_sp_times))")
println("∑|F_i - G_i|: $(sum(abs.(f_ls[1] - result.value[1])))")
println("Galley Log Exec: $(minimum(g_log_times))")
println("Galley Log Opt: $(minimum(g_log_opt_times))")
println("Finch (Dense) Log Exec: $(minimum(f_log_times))")
println("Finch (Sparse) Log Exec: $(minimum(f_sp_log_times))")
println("∑|F_i - G_i|: $(sum(abs.(f_logs[1] - result_log.value[1])))")
results = [("Algorithm", "Method", "ExecuteTime", "OptTime")]
push!(results, ("Linear Regression (SQ)", "Galley", string(minimum(g_times)), string(minimum(g_opt_times))))
push!(results, ("Linear Regression (SQ)", "Finch (Dense)", string(minimum(f_times)), string(0)))
push!(results, ("Linear Regression (SQ)", "Finch (Sparse)", string(minimum(f_sp_times)), string(0)))
push!(results, ("Logistic Regression (SQ)", "Galley", string(minimum(g_log_times)), string(minimum(g_log_opt_times))))
push!(results, ("Logistic Regression (SQ)", "Finch (Dense)", string(minimum(f_log_times)), string(0)))
push!(results, ("Logistic Regression (SQ)", "Finch (Sparse)", string(minimum(f_sp_log_times)), string(0)))
f_sp_logs = []
f_logs = []

function finch_sp_lr2(li_supp, supplier_x, li_part, part_x,  θ)
    X = Tensor(Dense(Sparse(Sparse(Element(0.0)))))
    X_idxs = Tensor(Dense(Sparse(Pattern())))
    li_supp_x = Tensor(Dense(Sparse(Element(0.0))))
    P = Tensor(Dense(SparseList(Element(0.0))))
    f_time = @elapsed @finch begin
        X .= 0
        X_idxs .= false
        for p = _
            for i1 =_
                for i2 =_
                    for x =_
                        X[x, i2, i1] += li_part[i1, p] * li_part[i2, p] * part_x[x, p]
                    end
                    if li_part[i1,p] > 0 && li_part[i2, p] > 0
                        X_idxs[i2, i1] = true
                    end
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
                    if X_idxs[i2, i1]
                        X[x, i2, i1] += li_supp_x[x, i1]
                    end
                end
            end
        end
    end
    println("Added Supp 1")
    f_time += @elapsed @finch begin
        for i1 =_
            for i2 =_
                for x =_
                    if X_idxs[i2, i1]
                        X[x, i2, i1] += li_supp_x[x, i2]
                    end
                end
            end
        end
    end
    println("Added Supp 2")
    f_time += @elapsed @finch begin
        P .= 0
        for i1=_
            for i2=_
                for j=_
                        P[i2, i1] += X[j, i2, i1] * θ[j]
                end
            end
        end
    end
    return f_time, P
end

# This formulation makes each row of the output a pair of line items for the same part
# and includes information about their suppliers
X_g = Materialize(t_undef, t_undef, t_undef,
                  :i1, :i2, :j,
                Σ(:p, MapJoin(*, Input(li_part, :i1, :p, "li_part"),
                                 Input(li_part, :i2, :p, "li_part"),
                                 MapJoin(+,
                                            Input(part_x, :j, :p, "part_x"),
                                            Σ(:s1, MapJoin(*, Input(li_supp, :i1, :s1, "li_supp"), Input(supplier_x, :j, :s1, "supplier_x"))),
                                            Σ(:s2, MapJoin(*, Input(li_supp, :i2, :s2, "li_supp"), Input(supplier_x, :j, :s2, "supplier_x")))))))

θ = Tensor(Dense(Element(0.0)), ones(Int, size(supplier_x)[1]) .% 100)
p_query = Query(:out, Materialize(t_undef, t_undef, :i1, :i2,  Σ(:k, MapJoin(*, X_g[:i1, :i2, :k], Input(θ, :k)))))
#insert_statistics!(DCStats, d_g)
g_times = []
g_opt_times = []
for i in 1:2
    result_galley = galley(p_query, ST=DCStats,  verbose=0)
    push!(g_times, result_galley.execute_time)
    push!(g_opt_times, result_galley.opt_time)
end
result = galley(p_query, ST=DCStats, verbose=3)

f_times = []
f_l = nothing
for i in 1:2
    f_time, P = finch_sp_lr2(li_supp, supplier_x, li_part, part_x, θ)
    push!(f_times, f_time)
    global f_l = P
    GC.gc()
end

# Need to be very careful about memory management to not go over 256 GB
function finch_sp_log2(li_supp, supplier_x, li_part, part_x, θ)
    X = Tensor(Dense(Sparse(Sparse(Element(0.0)))))
    X_idxs = Tensor(Dense(Sparse(Pattern())))
    li_supp_x = Tensor(Dense(Sparse(Element(0.0))))
    prediction_inter = Tensor(Dense(SparseList(Element(0.0))))
    prediction = Tensor(Dense(SparseList(Element(0.5))))

    f_time = @elapsed @finch begin
        X .= 0
        X_idxs .= false
        for p = _
            for i1 =_
                for i2 =_
                    for x =_
                        X[x, i2, i1] += li_part[i1, p] * li_part[i2, p] * part_x[x, p]
                    end
                    if li_part[i1,p] != 0 && li_part[i2, p] != 0
                        X_idxs[i2, i1] = true
                    end
                end
            end
        end
    end
    println("Initialized X and Y")
    println("nnz(X_idxs): $(countstored(X_idxs))")
    println("nnz(X): $(countstored(X))")
    println("size(X): $(size(X))")
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
    println("nnz(X): $(countstored(X))")
    println("size(X): $(size(X))")
    f_time += @elapsed @finch begin
        for i1 =_
            for i2 =_
                for x =_
                    if X_idxs[i2, i1]
                        X[x, i2, i1] += li_supp_x[x, i1]
                    end
                end
            end
        end
    end
    println("Added Supp 1")
    println("nnz(X): $(countstored(X))")
    println("size(X): $(size(X))")
    f_time += @elapsed @finch begin
        for i1 =_
            for i2 =_
                for x =_
                    if X_idxs[i2, i1]
                        X[x, i2, i1] += li_supp_x[x, i2]
                    end
                end
            end
        end
    end
    li_supp_x = Tensor(Dense(Sparse(Element(0.0))))
    GC.gc()
    println("Added Supp 2")
    println("nnz(X): $(countstored(X))")
    println("size(X): $(size(X))")
    f_time += @elapsed @finch begin
        prediction_inter .= 0
        for i1=_
            for i2=_
                for k=_
                    prediction_inter[i2, i1] += X[k, i2, i1] * θ[k]
                end
            end
        end
    end
    X = Tensor(Dense(Sparse(Sparse(Element(0.0)))))
    GC.gc()
    println("Made Predictions")
    println("nnz(prediction): $(countstored(prediction_inter))")
    println("size(prediction): $(size(prediction_inter))")
    f_time += @elapsed @finch begin
        prediction .= 0.5
        for i1=_
            for i2=_
                prediction[i2, i1] = sigmoid(prediction_inter[i2, i1])
            end
        end
    end
    return f_time, prediction
end

# Logistic Regression On Many-Many Join
p_g = Query(:P, Materialize(t_undef, t_undef, :i1, :i2, MapJoin(sigmoid, Σ(:k, MapJoin(*, X_g[:i1, :i2, :k], Input(θ, :k))))))
g_log_times = []
g_log_opt_times = []
for i in 1:2
    result_galley = galley(p_g, ST=DCStats,  verbose=0)
    push!(g_log_times, result_galley.execute_time)
    push!(g_log_opt_times, result_galley.opt_time)
end
result_log = galley(p_g, ST=DCStats, verbose=3)
GC.gc()

f_log_times = []
f_log = nothing
for i in 1:2
    f_time, P = finch_sp_log2(li_supp, supplier_x, li_part, part_x, θ)
    push!(f_log_times, f_time)
    global f_log = P
    GC.gc()
end

println("Galley Exec: $(minimum(g_times))")
println("Galley Opt: $(minimum(g_opt_times))")
println("Finch Exec: $(minimum(f_times))")
println("F = G: $(sum(abs.(f_l .- result.value)))")
println("Galley Log Exec: $(minimum(g_log_times))")
println("Galley Log Opt: $(minimum(g_log_opt_times))")
println("Finch Exec: $(minimum(f_log_times))")
println("F = G: $(sum(abs.(f_log .- result_log.value)))")

push!(results, ("Linear Regression (SJ)", "Galley", string(minimum(g_times)), string(minimum(g_opt_times))))
push!(results, ("Linear Regression (SJ)", "Finch (Sparse)", string(minimum(f_times)), string(0)))
push!(results, ("Logistic Regression (SJ)", "Galley", string(minimum(g_log_times)), string(minimum(g_log_opt_times))))
push!(results, ("Logistic Regression (SJ)", "Finch (Sparse)", string(minimum(f_log_times)), string(0)))
writedlm("Experiments/Results/tpch_inference.csv", results, ',')

using CSV
using DataFrames
using Measures
using CategoricalArrays
data = CSV.read("Experiments/Results/tpch_inference.csv", DataFrame)
data[!, :Speedup] = copy(data[!, :ExecuteTime])
for alg in unique(data.Algorithm)
    data[data.Algorithm .== alg, :Speedup] = data[(data.Algorithm .== alg) .& (data.Method .== "Finch (Sparse)"), :ExecuteTime] ./ data[data.Algorithm .== alg, :Speedup]
end
ordered_methods = CategoricalArray(data.Method)
levels!(ordered_methods, ["Galley", "Finch (Dense)", "Finch (Sparse)"])
gbplot = StatsPlots.groupedbar(data.Algorithm,
                                data.Speedup,
                                group = ordered_methods,
                                ylims=[.1, 50],
                                legend = :topleft,
                                size = (1400, 700),
                                ylabel = "Speedup",
                                xtickfontsize=15,
                                ytickfontsize=15,
                                xguidefontsize=16,
                                yguidefontsize=16,
                                legendfontsize=16,
                                left_margin=10mm)
savefig(gbplot, "Experiments/Figures/tpch_inference.png")
