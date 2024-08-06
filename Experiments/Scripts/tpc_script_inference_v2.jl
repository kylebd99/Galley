using Finch
using Galley: insert_statistics!, t_undef
using Galley
using DataFrames
using CSV
using SparseArrays
using Statistics
using Measures
using CategoricalArrays
using StatsFuns
include("../Experiments.jl")

relu(x) = max(0, x)

sigmoid(x) = 1.0 - exp(x-log1pexp(x))

function simplify_col(df, x, val_to_idx)
    df[!, x] = [val_to_idx[x] for x in df[!, x]]
end

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

function cols_to_join_tensor(df, xs, dims)
    dims = map(Int64, dims)
    Is = [Int64[] for _ in xs]
    V = Int64[]
    for row in eachrow(df)
        for (i, x) in enumerate(xs)
            push!(Is[i], row[x])
        end
        push!(V, 1)
    end
    tensor = Element(0)
    for _ in 1:length(xs)-1
        tensor = SparseList(tensor)
    end
    tensor = Dense(tensor)
    return Tensor(tensor, fsparse(Is..., V, dims))
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

function finch_lr(li_tns, orders_x, order_cust, customer_x, supplier_x, part_x, θ; dense=false)
    X = dense ? Tensor(Dense(Dense(Element(0.0)))) : Tensor(Dense(Sparse(Element(0.0))))
    P = Tensor(Dense((Element(0.0))))
    f_time = @elapsed @finch begin
        X .= 0
        for i =_
            for o=_
                for s=_
                    for p=_
                        for c=_
                            for j =_
                                X[j, i] += li_tns[p, s, o, i] * order_cust[c, o] * (orders_x[j, o] + customer_x[j, c])
                            end
                        end
                    end
                end
            end
        end
    end
    println("First X Step")
    f_time += @elapsed @finch begin
        for i =_
            for o=_
                for s=_
                    for p=_
                        for j =_
                            X[j, i] += li_tns[p, s, o, i] * (supplier_x[j, s] + part_x[j, p])
                        end
                    end
                end
            end
        end
    end
    println("Built X")
    f_time += @elapsed @finch begin
        P .= 0
        for i=_
            for j=_
                P[i] +=  X[j, i] * θ[j]
            end
        end
    end
    return f_time, P
end

function finch_log(li_tns, orders_x, order_cust, customer_x, supplier_x, part_x, θ; dense=false)
    X = dense ? Tensor(Dense(Dense(Element(0.0)))) : Tensor(Dense(Sparse(Element(0.0))))
    p_inter = Tensor(Dense(Element(0.0)))
    prediction = Tensor(Dense(Element(0.5)))
    f_time = @elapsed @finch begin
        X .= 0
        for i =_
            for o=_
                for s=_
                    for p=_
                        for c=_
                            for j =_
                                X[j, i] += li_tns[p, s, o, i] * order_cust[c, o] * (orders_x[j, o] + customer_x[j, c])
                            end
                        end
                    end
                end
            end
        end
    end
    println("First X Step")
    f_time += @elapsed @finch begin
        for i =_
            for o=_
                for s=_
                    for p=_
                        for j =_
                            X[j, i] += li_tns[p, s, o, i] * (supplier_x[j, s] + part_x[j, p])
                        end
                    end
                end
            end
        end
    end
    println("Built X")
    f_time += @elapsed @finch begin
        p_inter .= 0
        for i=_
            for j=_
                p_inter[i] += X[j, i] * θ[j]
            end
        end

        prediction .= 0.5
        for i = _
            prediction[i] = sigmoid(p_inter[i])
        end
    end
    return f_time, prediction
end

function finch_nn(li_tns, orders_x, order_cust, customer_x, supplier_x, part_x, W1, W2; dense=false)
    X = dense ? Tensor(Dense(Dense(Element(0.0)))) : Tensor(Dense(Sparse(Element(0.0))))
    h1 = Tensor(Dense(Dense(Element(0.0))))
    h1_relu = Tensor(Dense(Dense(Element(0.0))))
    h2 = Tensor(Dense(Element(0.0)))
    prediction = Tensor(Dense(Element(0.5)))
    f_time = @elapsed @finch begin
        X .= 0
        for i =_
            for o=_
                for s=_
                    for p=_
                        for c=_
                            for j =_
                                X[j, i] += li_tns[p, s, o, i] * order_cust[c, o] * (orders_x[j, o] + customer_x[j, c])
                            end
                        end
                    end
                end
            end
        end
    end
    println("First X Step")
    f_time += @elapsed @finch begin
        for i =_
            for o=_
                for s=_
                    for p=_
                        for j =_
                            X[j, i] += li_tns[p, s, o, i] * (supplier_x[j, s] + part_x[j, p])
                        end
                    end
                end
            end
        end
    end
    println("Built X")
    f_time += @elapsed @finch begin
        h1 .= 0
        for i=_
            for j=_
                for k=_
                    h1[k, i] += X[j, i] * W1[k, j]
                end
            end
        end
    end

    f_time += @elapsed @finch begin
        h1_relu .= 0
        for i=_
            for k=_
                h1_relu[k, i] = relu(h1[k, i])
            end
        end
    end

    f_time += @elapsed @finch begin
        h2 .= 0
        for i=_
            for j=_
                h2[i] += h1_relu[j, i] * W2[j]
            end
        end
    end

    f_time += @elapsed @finch begin
        prediction .= 0.5
        for i = _
            prediction[i] = sigmoid(h2[i])
        end
    end
    return f_time, prediction
end

function finch_cov(li_tns, orders_x, order_cust, customer_x, supplier_x, part_x; dense=false)
    X = dense ? Tensor(Dense(Dense(Element(0.0)))) : Tensor(Dense(Sparse(Element(0.0))))
    cov = Tensor(Dense(Dense(Element(0.0))))
    f_time = @elapsed @finch begin
        X .= 0
        for i =_
            for o=_
                for s=_
                    for p=_
                        for c=_
                            for j =_
                                X[j, i] += li_tns[p, s, o, i] * order_cust[c, o] * (orders_x[j, o] + customer_x[j, c])
                            end
                        end
                    end
                end
            end
        end
    end
    println("First X Step")
    f_time += @elapsed @finch begin
        for i =_
            for o=_
                for s=_
                    for p=_
                        for j =_
                            X[j, i] += li_tns[p, s, o, i] * (supplier_x[j, s] + part_x[j, p])
                        end
                    end
                end
            end
        end
    end
    println("Built X")
    f_time += @elapsed @finch begin
        cov .= 0
        for i=_
            for j=_
                for k=_
                    cov[k, j] += X[j, i] * X[k, i]
                end
            end
        end
    end

    return f_time, cov
end

function finch_sp_lr2(li_tns, supplier_x, part_x,  θ; dense=false)
    X = dense ? Tensor(Dense(Sparse(Sparse(Element(0.0))))) : Tensor(Dense(Sparse(Sparse(Element(0.0)))))
    P = Tensor(Dense(SparseList(Element(0.0))))
    f_time = @elapsed @finch begin
        X .= 0
        for p = _
            for i1 =_
                for i2 =_
                    for s1 =_
                        for s2 =_
                            for j =_
                                X[j, i2, i1] += li_tns[s1, i1, p] * li_tns[s2, i2, p] * (part_x[j, p] + supplier_x[j, s1])
                            end
                        end
                    end
                end
            end
        end
    end
    f_time += @elapsed @finch begin
        for p = _
            for i1 =_
                for i2 =_
                    for s1 =_
                        for s2 =_
                            for j =_
                                X[j, i2, i1] += li_tns[s1, i1, p] * li_tns[s2, i2, p] * supplier_x[j, s2]
                            end
                        end
                    end
                end
            end
        end
    end
    println("Built X")
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

function finch_sp_log2(li_tns, supplier_x, part_x, θ)
    X = Tensor(Dense(Sparse(Sparse(Element(0.0)))))
    prediction_inter = Tensor(Dense(SparseList(Element(0.0))))
    prediction = Tensor(Dense(SparseList(Element(0.5))))
    f_time = @elapsed @finch begin
        X .= 0
        for p = _
            for i1 =_
                for i2 =_
                    for s1 =_
                        for s2 =_
                            for j =_
                                X[j, i2, i1] += li_tns[s1, i1, p] * li_tns[s2, i2, p] *(part_x[j, p] + supplier_x[j, s1] + supplier_x[j, s2])
                            end
                        end
                    end
                end
            end
        end
    end
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

function finch_sp_cov2(li_tns, supplier_x, part_x)
    X = Tensor(Dense(Sparse(Sparse(Element(0.0)))))
    cov = Tensor(Dense(Dense(Element(0.0))))
    f_time = @elapsed @finch begin
        X .= 0
        for p = _
            for i1 =_
                for i2 =_
                    for s1 =_
                        for s2 =_
                            for j =_
                                X[j, i2, i1] += li_tns[s1, i1, p] * li_tns[s2, i2, p] *(part_x[j, p] + supplier_x[j, s1] + supplier_x[j, s2])
                            end
                        end
                    end
                end
            end
        end
    end
    println("nnz(X): $(countstored(X))")
    println("size(X): $(size(X))")
    f_time += @elapsed @finch begin
        cov .= 0
        for i1=_
            for i2=_
                for j=_
                    for k=_
                        cov[j, k] += X[k, i2, i1] * X[j, i2, i1]
                    end
                end
            end
        end
    end
    return f_time, cov
end

function main()
    # First, we load the TPCH data
    customer = CSV.read("Experiments/Data/TPCH/customer.tbl", DataFrame, delim='|', header=[:CustomerKey, :Name, :Address, :NationKey, :Phone, :AcctBal, :MktSegment, :Comment, :Col9])
    lineitem = CSV.read("Experiments/Data/TPCH/lineitem.tbl", DataFrame, delim='|', header=[:OrderKey, :PartKey, :SuppKey, :LineNumber, :Quantity, :ExtendedPrice, :Discount, :Tax, :ReturnFlag, :LineStatus, :ShipDate, :CommitDate, :ReceiptDate, :ShipInstruct, :ShipMode, :Comment])
    orders = CSV.read("Experiments/Data/TPCH/orders.tbl", DataFrame, delim='|', header=[:OrderKey, :CustomerKey, :OrderStatus, :TotalPrice, :OrderDate, :OrderPriority, :Clerk, :ShipPriority, :Comment])
    partsupp = CSV.read("Experiments/Data/TPCH/partsupp.tbl", DataFrame, delim='|', header=[:PartKey, :SuppKey, :AvailQty, :SupplyCost, :Comment])
    part = CSV.read("Experiments/Data/TPCH/part.tbl", DataFrame, delim='|', header=[:PartKey, :Name, :MFGR, :Brand, :Type, :Size, :Container, :RetailPrice, :Comment])
    supplier = CSV.read("Experiments/Data/TPCH/supplier.tbl", DataFrame, delim='|', header=[:SuppKey, :Name, :Address, :NationKey, :Phone, :AcctBal, :Comment])
    lineitem[!, :LineItemKey] = 1:nrow(lineitem)
    orderkey_idx = Dict(o => i for (i, o) in enumerate(unique(orders.OrderKey)))
    partkey_idx = Dict(p => i for (i, p) in enumerate(unique(part.PartKey)))
    suppkey_idx = Dict(s => i for (i, s) in enumerate(unique(supplier.SuppKey)))
    customerkey_idx = Dict(c => i for (i, c) in enumerate(unique(customer.CustomerKey)))

    simplify_col(lineitem, :OrderKey, orderkey_idx)
    simplify_col(lineitem, :PartKey, partkey_idx)
    simplify_col(lineitem, :SuppKey, suppkey_idx)
    simplify_col(orders, :OrderKey, orderkey_idx)
    simplify_col(orders, :CustomerKey, customerkey_idx)
    simplify_col(customer, :CustomerKey, customerkey_idx)
    simplify_col(part, :PartKey, partkey_idx)
    simplify_col(supplier, :SuppKey, suppkey_idx)

    # Conceptually, each row of X is going to be a line item.
    lineitem = lineitem[!, [:LineItemKey, :OrderKey, :PartKey, :SuppKey, :LineNumber, :Quantity, :ExtendedPrice, :Discount, :Tax, :ReturnFlag, :LineStatus, :ShipMode]]
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
    lineitem_y = Vector(lineitem[!, :ExtendedPrice])
    li_tns = cols_to_join_tensor(lineitem, (:PartKey, :SuppKey, :OrderKey,  :LineItemKey), (maximum(values(partkey_idx)), maximum(values(suppkey_idx)), maximum(values(orderkey_idx)), nrow(lineitem)))
    li_tns2 = cols_to_join_tensor(lineitem, (:SuppKey, :LineItemKey, :PartKey), (maximum(values(suppkey_idx)), nrow(lineitem), maximum(values(partkey_idx))))
    orders_x = Matrix(select(orders, Not([:OrderKey, :CustomerKey])))'
    order_cust = cols_to_join_tensor(orders, (:CustomerKey, :OrderKey), (maximum(values(customerkey_idx)), maximum(values(orderkey_idx))))
    customer_x = Matrix(select(customer, Not([:CustomerKey])))'
    supplier_x = Matrix(select(supplier, Not([:SuppKey])))'
    part_x = Matrix(select(part, Not([:PartKey])))'

    x_starts = cumsum([0, size(orders_x)[1], size(customer_x)[1], size(supplier_x)[1], size(part_x)[1]])
    x_dim = x_starts[end]
    orders_x = align_x_dims(orders_x, x_starts[1], x_dim)
    customer_x = align_x_dims(customer_x, x_starts[2], x_dim)
    supplier_x = align_x_dims(supplier_x, x_starts[3], x_dim)
    part_x = align_x_dims(part_x, x_starts[4], x_dim)

    n_reps = 3
    X_g = Mat(:i, :j, Σ(:o, :s, :p, MapJoin(*,  Input(li_tns, :p, :s, :o, :i, "li_tns"),
                                                MapJoin(+, Input(orders_x, :j, :o, "orders_x"),
                                                            Σ(:c, MapJoin(*, Input(order_cust, :c, :o, "order_cust"),
                                                                             Input(customer_x, :j, :c, "customer_x"))),
                                                            Input(supplier_x, :j, :s, "supplier_x"),
                                                            Input(part_x, :j, :p, "part_x")))))
    insert_statistics!(DCStats, X_g)
    θ = Tensor(Dense(Element(0.0)), ones(Int, size(supplier_x)[1]) .% 100)
    P_query = Query(:out, Mat(:i, Aggregate(+, :j, MapJoin(*, X_g[:i, :j], Input(θ, :j)))))
    g_times = []
    g_opt_times = []
    result = galley(P_query, faq_optimizer=greedy, ST=DCStats, verbose=3)
    for _ in 1:n_reps
        result_galley = galley(P_query, faq_optimizer=greedy, ST=DCStats,  verbose=0)
        push!(g_times, result_galley.execute_time)
        push!(g_opt_times, result_galley.opt_time)
    end

    f_times = []
    f_ls = []
    for _ in 1:n_reps
        f_time, p = finch_lr(li_tns, orders_x, order_cust, customer_x, supplier_x, part_x, θ; dense=true)
        push!(f_times, f_time)
        push!(f_ls, p)
    end
    f_sp_times = []
    f_sp_ls = []
    for _ in 1:n_reps
        f_time, p = finch_lr(li_tns, orders_x, order_cust, customer_x, supplier_x, part_x, θ; dense=false)
        push!(f_sp_times, f_time)
        push!(f_sp_ls, p)
    end

    p_query = Query(:out, Mat(:i, MapJoin(sigmoid, Σ(:k, MapJoin(*, X_g[:i, :k], Input(θ, :k))))))
    g_log_times = []
    g_log_opt_times = []
    for _ in 1:n_reps
        result_galley = galley(p_query, faq_optimizer=greedy, ST=DCStats,  verbose=0)
        push!(g_log_times, result_galley.execute_time)
        push!(g_log_opt_times, result_galley.opt_time)
    end
    result_log = galley(p_query, faq_optimizer=greedy, ST=DCStats, verbose=3)

    f_log_times = []
    f_logs = []
    for _ in 1:n_reps
        f_time, prediction = finch_log(li_tns, orders_x, order_cust, customer_x, supplier_x, part_x, θ; dense=true)
        push!(f_log_times, f_time)
        push!(f_logs, prediction)
    end

    f_sp_log_times = []
    f_sp_logs = []
    for _ in 1:n_reps
        f_time, prediction = finch_log(li_tns, orders_x, order_cust, customer_x, supplier_x, part_x, θ; dense=false)
        push!(f_sp_log_times, f_time)
        push!(f_sp_logs, prediction)
    end

    hidden_layer_size = 10
    feature_size = size(part_x)[1]
    W1 = Tensor(Dense(Dense(Element(0.0))), rand(hidden_layer_size, feature_size))
    h1 = Mat(:i, :k1, MapJoin(relu, Σ(:j, MapJoin(*, X_g[:i, :j], Input(W1, :k1, :j)))))
    W2 = Tensor(Dense(Element(0.0)), rand(hidden_layer_size))
    h2 = Mat(:i, MapJoin(sigmoid, Σ(:k1, MapJoin(*, h1[:i, :k1], Input(W2, :k1)))))
    P = Query(:out, h2)
    g_nn_times = []
    g_nn_opt_times = []
    for _ in 1:n_reps
        result_galley = galley(P, faq_optimizer=greedy, ST=DCStats,  verbose=0)
        push!(g_nn_times, result_galley.execute_time)
        push!(g_nn_opt_times, result_galley.opt_time)
    end
    result_nn = galley(P, faq_optimizer=greedy, ST=DCStats, verbose=3)

    f_nn_times = []
    f_nns = []
    for _ in 1:n_reps
        f_time, prediction = finch_nn(li_tns, orders_x, order_cust, customer_x, supplier_x, part_x, W1, W2; dense=true)
        push!(f_nn_times, f_time)
        push!(f_nns, prediction)
    end

    f_sp_nn_times = []
    f_sp_nns = []
    for _ in 1:n_reps
        f_time, prediction = finch_nn(li_tns, orders_x, order_cust, customer_x, supplier_x, part_x, W1, W2; dense=false)
        push!(f_sp_nn_times, f_time)
        push!(f_sp_nns, prediction)
    end
    P = Query(:out, Mat(:j, :k, Σ(:i, MapJoin(*, X_g[:i, :j], X_g[:i, :k]))))
    g_cov_times = []
    g_cov_opt_times = []
    for _ in 1:n_reps
        result_galley = galley(P, faq_optimizer=greedy, ST=DCStats,  verbose=3)
        push!(g_cov_times, result_galley.execute_time)
        push!(g_cov_opt_times, result_galley.opt_time)
    end
    result_cov = galley(P, faq_optimizer=greedy, ST=DCStats, verbose=3)

    f_cov_times = []
    f_covs = []
    for _ in 1:n_reps
        f_time, prediction = finch_cov(li_tns, orders_x, order_cust, customer_x, supplier_x, part_x; dense=true)
        push!(f_cov_times, f_time)
        push!(f_covs, prediction)
    end

    f_sp_cov_times = []
    f_sp_covs = []
    for _ in 1:n_reps
        f_time, prediction = finch_cov(li_tns, orders_x, order_cust, customer_x, supplier_x, part_x; dense=false)
        push!(f_sp_cov_times, f_time)
        push!(f_sp_covs, prediction)
    end

     println("Galley Exec: $(minimum(g_times))")
    println("Galley Opt: $(minimum(g_opt_times))")
    println("Finch (Dense) Exec: $(minimum(f_times))")
    println("Finch (Sparse) Exec: $(minimum(f_sp_times))")
    println("∑|F_i - G_i|: $(sum(abs.(f_ls[1] - result.value)))")
    println("Galley Log Exec: $(minimum(g_log_times))")
    println("Galley Log Opt: $(minimum(g_log_opt_times))")
    println("Finch (Dense) Log Exec: $(minimum(f_log_times))")
    println("Finch (Sparse) Log Exec: $(minimum(f_sp_log_times))")
    println("∑|F_i - G_i|: $(sum(abs.(f_logs[1] - result_log.value)))")
    println("Galley NN Exec: $(minimum(g_nn_times))")
    println("Galley NN Opt: $(minimum(g_nn_opt_times))")
    println("Finch (Dense) NN Exec: $(minimum(f_nn_times))")
    println("Finch (Sparse) NN Exec: $(minimum(f_sp_nn_times))")
    println("∑|F_i - G_i|: $(sum(abs.(f_nns[1] - result_nn.value)))")
    println("Galley Cov Exec: $(minimum(g_cov_times))")
    println("Galley Cov Opt: $(minimum(g_cov_opt_times))")
    println("Finch (Dense) Cov Exec: $(minimum(f_cov_times))")
    println("Finch (Sparse) Cov Exec: $(minimum(f_sp_cov_times))")
    println("∑|F_i - G_i|: $(sum(abs.(f_covs[1] - result_cov.value)))")

    f_ls = []
    f_sp_ls = []
    f_logs = []
    f_sp_logs = []
    f_nns = []
    results = [("Algorithm", "Method", "ExecuteTime", "OptTime")]
    push!(results, ("Linear Regression (SQ)", "Galley", string(minimum(g_times)), string(minimum(g_opt_times))))
    push!(results, ("Linear Regression (SQ)", "Finch (Dense)", string(minimum(f_times)), string(0)))
    push!(results, ("Linear Regression (SQ)", "Finch (Sparse)", string(minimum(f_sp_times)), string(0)))
    push!(results, ("Logistic Regression (SQ)", "Galley", string(minimum(g_log_times)), string(minimum(g_log_opt_times))))
    push!(results, ("Logistic Regression (SQ)", "Finch (Dense)", string(minimum(f_log_times)), string(0)))
    push!(results, ("Logistic Regression (SQ)", "Finch (Sparse)", string(minimum(f_sp_log_times)), string(0)))
    push!(results, ("Neural Network (SQ)", "Galley", string(minimum(g_nn_times)), string(minimum(g_nn_opt_times))))
    push!(results, ("Neural Network (SQ)", "Finch (Dense)", string(minimum(f_nn_times)), string(0)))
    push!(results, ("Neural Network (SQ)", "Finch (Sparse)", string(minimum(f_sp_nn_times)), string(0)))
    push!(results, ("Covariance (SQ)", "Galley", string(minimum(g_cov_times)), string(minimum(g_cov_opt_times))))
    push!(results, ("Covariance (SQ)", "Finch (Dense)", string(minimum(f_cov_times)), string(0)))
    push!(results, ("Covariance (SQ)", "Finch (Sparse)", string(minimum(f_sp_cov_times)), string(0)))

    # This formulation makes each row of the output a pair of line items for the same part
    # and includes information about their suppliers li_tns[s1, i1, p]
    X_g = Mat(:i1, :i2, :j,
              Σ(:p, :s1, :s2, MapJoin(*, Input(li_tns2, :s1, :i1, :p, "li_part"),
                                        Input(li_tns2, :s2, :i2, :p, "li_part"),
                                        MapJoin(+,
                                                    Input(part_x, :j, :p, "part_x"),
                                                    Input(supplier_x, :j, :s1, "supplier_x"),
                                                    Input(supplier_x, :j, :s2, "supplier_x")))))
    insert_statistics!(DCStats, X_g)
    θ = Tensor(Dense(Element(0.0)), ones(Int, size(supplier_x)[1]) .% 100)
    p_query = Query(:out, Materialize(t_undef, t_undef, :i1, :i2,  Σ(:k, MapJoin(*, X_g[:i1, :i2, :k], Input(θ, :k)))))
    g_times = []
    g_opt_times = []
    for _ in 1:n_reps
        result_galley = galley(p_query, ST=DCStats, faq_optimizer=greedy,  verbose=0)
        push!(g_times, result_galley.execute_time)
        push!(g_opt_times, result_galley.opt_time)
    end
    result = galley(p_query, ST=DCStats, faq_optimizer=greedy, verbose=3)

    f_times = []
    f_l = nothing
    for _ in 1:n_reps
        f_time, P = finch_sp_lr2(li_tns2, supplier_x, part_x, θ)
        push!(f_times, f_time)
        f_l = P
    end

    # Logistic Regression On Many-Many Join
    p_g = Query(:P, Materialize(t_undef, t_undef, :i1, :i2, MapJoin(sigmoid, Σ(:k, MapJoin(*, X_g[:i1, :i2, :k], Input(θ, :k))))))
    g_log_times = []
    g_log_opt_times = []
    for _ in 1:n_reps
        result_galley = galley(p_g, ST=DCStats, faq_optimizer=greedy, verbose=0)
        push!(g_log_times, result_galley.execute_time)
        push!(g_log_opt_times, result_galley.opt_time)
    end
    result_log = galley(p_g, ST=DCStats, faq_optimizer=greedy, verbose=3)

    f_log_times = []
    f_log = nothing
    for _ in 1:n_reps
        f_time, P = finch_sp_log2(li_tns2, supplier_x, part_x, θ)
        push!(f_log_times, f_time)
        f_log = P
    end

    # Logistic Regression On Many-Many Join
    p_g = Query(:P, Materialize(t_undef, t_undef, :j, :k, Σ(:i1, :i2, MapJoin(*, X_g[:i1, :i2, :k], X_g[:i1, :i2, :j]))))
    g_log_times = []
    g_log_opt_times = []
    for _ in 1:n_reps
        result_galley = galley(p_g, ST=DCStats, faq_optimizer=greedy, verbose=0)
        push!(g_log_times, result_galley.execute_time)
        push!(g_log_opt_times, result_galley.opt_time)
    end
    result_log = galley(p_g, ST=DCStats, faq_optimizer=greedy, verbose=3)

    f_cov_times = []
    f_cov = nothing
    for _ in 1:n_reps
        f_time, P = finch_sp_cov2(li_tns2, supplier_x, part_x)
        push!(f_cov_times, f_time)
        f_cov = P
    end
    println("Galley Exec: $(minimum(g_times))")
    println("Galley Opt: $(minimum(g_opt_times))")
    println("Finch Exec: $(minimum(f_times))")
    println("F = G: $(sum(abs.(f_l .- result.value)))")
    println("Galley Log Exec: $(minimum(g_log_times))")
    println("Galley Log Opt: $(minimum(g_log_opt_times))")
    println("Finch Log Exec: $(minimum(f_log_times))")
    println("F = G: $(sum(abs.(f_log .- result_log.value)))")
    println("Galley Cov Exec: $(minimum(g_cov_times))")
    println("Galley Cov Opt: $(minimum(g_cov_opt_times))")
    println("Finch Cov Exec: $(minimum(f_cov_times))")
    println("F = G: $(sum(abs.(f_cov .- result_cov.value)))")

    push!(results, ("Linear Regression (SJ)", "Galley", string(minimum(g_times)), string(minimum(g_opt_times))))
    push!(results, ("Linear Regression (SJ)", "Finch (Sparse)", string(minimum(f_times)), string(0)))
    push!(results, ("Logistic Regression (SJ)", "Galley", string(minimum(g_log_times)), string(minimum(g_log_opt_times))))
    push!(results, ("Logistic Regression (SJ)", "Finch (Sparse)", string(minimum(f_log_times)), string(0)))
    push!(results, ("Covariance (SJ)", "Galley", string(minimum(g_cov_times)), string(minimum(g_cov_opt_times))))
    push!(results, ("Covariance (SJ)", "Finch (Sparse)", string(minimum(f_cov_times)), string(0)))

    writedlm("Experiments/Results/tpch_inference.csv", results, ',')
    data = CSV.read("Experiments/Results/tpch_inference.csv", DataFrame)
    data[!, :Speedup] = copy(data[!, :ExecuteTime])
    for alg in unique(data.Algorithm)
        if length(data[(data.Algorithm .== alg) .& (data.Method .== "Finch (Sparse)"), :ExecuteTime]) > 0
            data[data.Algorithm .== alg, :Speedup] = data[(data.Algorithm .== alg) .& (data.Method .== "Finch (Sparse)"), :ExecuteTime] ./ data[data.Algorithm .== alg, :Speedup]
        else
            data[data.Algorithm .== alg, :Speedup] = data[(data.Algorithm .== alg) .& (data.Method .== "Finch (Dense)"), :ExecuteTime] ./ data[data.Algorithm .== alg, :Speedup]
        end
    end
    data[!, :Speedup] = log10.(data.Speedup)
    ordered_methods = CategoricalArray(data.Method)
    levels!(ordered_methods, ["Galley", "Finch (Dense)", "Finch (Sparse)"])
    gbplot = StatsPlots.groupedbar(data.Algorithm,
                                    data.Speedup,
                                    group = ordered_methods,
                                    legend = :topright,
                                    size = (1800, 700),
                                    ylabel = "Speedup (10^x)",
                                    ylims=[-.5, 2.5],
                                    xtickfontsize=15,
                                    ytickfontsize=15,
                                    xguidefontsize=16,
                                    yguidefontsize=16,
                                    legendfontsize=16,
                                    left_margin=15mm,
                                    bottom_margin=10mm,
                                    fillrange=-1)
    savefig(gbplot, "Experiments/Figures/tpch_inference.png")
end

main()
