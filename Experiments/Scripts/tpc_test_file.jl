using Finch
using Galley: insert_statistics!, t_undef, fill_table
using Galley
using SparseArrays
using Statistics
using StatsFuns
using Profile
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
    li_tns = cols_to_join_tensor(lineitem, (:PartKey, :SuppKey, :OrderKey,  :LineItemKey), (maximum(values(partkey_idx)), maximum(values(suppkey_idx)), maximum(values(orderkey_idx)), nrow(lineitem)))
    li_tns2 = cols_to_join_tensor(lineitem, (:SuppKey, :LineItemKey, :PartKey), (maximum(values(suppkey_idx)), nrow(lineitem), maximum(values(partkey_idx))))
    orders_x = floor.(Matrix(select(orders, Not([:OrderKey, :CustomerKey])))') .% 100
    order_cust = cols_to_join_tensor(orders, (:CustomerKey, :OrderKey), (maximum(values(customerkey_idx)), maximum(values(orderkey_idx))))
    customer_x = floor.(Matrix(select(customer, Not([:CustomerKey])))') .% 100
    supplier_x = floor.(Matrix(select(supplier, Not([:SuppKey])))') .% 100
    part_x = floor.(Matrix(select(part, Not([:PartKey])))') .% 100

    x_starts = cumsum([0, size(orders_x)[1], size(customer_x)[1], size(supplier_x)[1], size(part_x)[1]])
    x_dim = x_starts[end]
    orders_x = align_x_dims(orders_x, x_starts[1], x_dim)
    customer_x = align_x_dims(customer_x, x_starts[2], x_dim)
    supplier_x = align_x_dims(supplier_x, x_starts[3], x_dim)
    part_x = align_x_dims(part_x, x_starts[4], x_dim)
    n_reps = 5
    optimizer = pruned
    X_g = Mat(:i, :j, Σ(:o, :s, :p, MapJoin(*,  Input(li_tns, :p, :s, :o, :i, "li_tns"),
                                                MapJoin(+, Input(orders_x, :j, :o, "orders_x"),
                                                            Σ(:c, MapJoin(*, Input(order_cust, :c, :o, "order_cust"),
                                                                             Input(customer_x, :j, :c, "customer_x"))),
                                                            Input(supplier_x, :j, :s, "supplier_x"),
                                                            Input(part_x, :j, :p, "part_x")))))
    insert_statistics!(DCStats, X_g)
    θ = Tensor(Dense(Element(0)), ones(Int, size(supplier_x)[1]) .% 100)

    # ---------------- Linear Regression On Star Join -------------------
    P_query = Query(:out, Mat(:i, Aggregate(+, :j, MapJoin(*, X_g[:i, :j], Input(θ, :j)))))
    g_times = []
    g_opt_times = []
    for _ in 1:n_reps
        result_galley = galley(P_query, faq_optimizer=optimizer, ST=DCStats,  verbose=0)
        push!(g_times, result_galley.execute_time)
        push!(g_opt_times, result_galley.opt_time)
    end
    result = galley(P_query, faq_optimizer=optimizer, ST=DCStats, verbose=3)
    # ---------------- Logistic Regression On Star Join -------------------
    p_query = Query(:out, Mat(:i, MapJoin(sigmoid, Σ(:k, MapJoin(*, X_g[:i, :k], Input(θ, :k))))))
    g_log_times = []
    g_log_opt_times = []
    for _ in 1:n_reps
        result_galley = galley(p_query, faq_optimizer=optimizer, ST=DCStats, verbose=0)
        push!(g_log_times, result_galley.execute_time)
        push!(g_log_opt_times, result_galley.opt_time)
    end
    result_log = galley(p_query, faq_optimizer=optimizer, ST=DCStats, verbose=3)

    # ---------------- Neural Network Inference On Star Join -------------------
    hidden_layer_size = 25
    feature_size = size(part_x)[1]
    W1 = Tensor(Dense(Dense(Element(0))), rand(Int, hidden_layer_size, feature_size) .% 10)
    h1 = Mat(:i, :k1, MapJoin(relu, Σ(:j, MapJoin(*, X_g[:i, :j], Input(W1, :k1, :j)))))
    W2 = Tensor(Dense(Dense(Element(0))), rand(Int, hidden_layer_size, hidden_layer_size) .% 10)
    h2 = Mat(:i, :k2, MapJoin(relu, Σ(:k1, MapJoin(*, h1[:i, :k1], Input(W2, :k2, :k1)))))
    W3 = Tensor(Dense(Element(0)), rand(Int, hidden_layer_size) .% 10)
    h3 = Mat(:i, MapJoin(sigmoid, Σ(:k2, MapJoin(*, h2[:i, :k2], Input(W3, :k2)))))
    P = Query(:out, h3)
    g_nn_times = []
    g_nn_opt_times = []
    for _ in 1:n_reps
        result_galley = galley(P, faq_optimizer=optimizer, ST=DCStats,  verbose=0)
        push!(g_nn_times, result_galley.execute_time)
        push!(g_nn_opt_times, result_galley.opt_time)
    end
    result_nn = galley(P, faq_optimizer=optimizer, ST=DCStats, verbose=3)

    # ---------------- Covariance Matrix Computation On Star Join -------------------
    P = Query(:out, Mat(:j, :k, Σ(:i, MapJoin(*, X_g[:i, :j], X_g[:i, :k]))))
    g_cov_times = []
    g_cov_opt_times = []
    for _ in 1:n_reps
        P = Query(:out, Mat(:j, :k, Σ(:i, MapJoin(*, X_g[:i, :j], X_g[:i, :k]))))
        result_galley = galley(P, faq_optimizer=optimizer, ST=DCStats,  verbose=3)
        push!(g_cov_times, result_galley.execute_time)
        push!(g_cov_opt_times, result_galley.opt_time)
    end
    result_cov = galley(P, faq_optimizer=optimizer, ST=DCStats, verbose=3)

    # TODO: print out the median, min and max to get a sense of variance
    println("Galley Exec: $(mean(g_times[2:end]))")
    println("Galley Opt: $(mean(g_opt_times[2:end]))")
    println("Galley Log Exec: $(mean(g_log_times[2:end]))")
    println("Galley Log Opt: $(mean(g_log_opt_times[2:end]))")
    println("Galley NN Exec: $(mean(g_nn_times[2:end]))")
    println("Galley NN Opt: $(mean(g_nn_opt_times[2:end]))")
    println("Galley Cov Exec: $(mean(g_cov_times[2:end]))")
    println("Galley Cov Opt: $(mean(g_cov_opt_times[2:end]))")

    # This formulation makes each row of the output a pair of line items for the same part
    # and includes information about their suppliers li_tns[s1, i1, p]
    supplier_x = floor.(Matrix(select(supplier, Not([:SuppKey])))') .% 100
    part_x = floor.(Matrix(select(part, Not([:PartKey])))') .% 100
    x_starts = cumsum([0, size(supplier_x)[1], size(supplier_x)[1], size(part_x)[1]])
    x_dim = x_starts[end]
    supplier_x1 = align_x_dims(supplier_x, x_starts[1], x_dim)
    supplier_x2 = align_x_dims(supplier_x, x_starts[2], x_dim)
    part_x = align_x_dims(part_x, x_starts[3], x_dim)

    X_g = Mat(:i1, :i2, :j,
              Σ(:p, :s1, :s2, MapJoin(*,Input(li_tns2, :s1, :i1, :p, "li_part"),
                                        Input(li_tns2, :s2, :i2, :p, "li_part"),
                                        MapJoin(+,
                                                    Input(part_x, :j, :p, "part_x"),
                                                    Input(supplier_x1, :j, :s1, "supplier_x1"),
                                                    Input(supplier_x2, :j, :s2, "supplier_x2")))))
    insert_statistics!(DCStats, X_g)
    θ = Tensor(Dense(Element(0.0)), ones(Int, size(supplier_x1)[1]) .% 100)

    # ---------------- Linear Regression On Many-Many Join -------------------
    p_query = Query(:out, Materialize(t_undef, t_undef, :i1, :i2,  Σ(:k, MapJoin(*, X_g[:i1, :i2, :k], Input(θ, :k)))))
    g_times = []
    g_opt_times = []
    for _ in 1:n_reps
        result_galley = galley(p_query, ST=DCStats, faq_optimizer=optimizer,  verbose=0)
        push!(g_times, result_galley.execute_time)
        push!(g_opt_times, result_galley.opt_time)
    end
    result = galley(p_query, ST=DCStats, faq_optimizer=optimizer, verbose=3)

    # ---------------- Logistic Regression On Many-Many Join -------------------
    p_g = Query(:P, Materialize(t_undef, t_undef, :i1, :i2, MapJoin(sigmoid, Σ(:k, MapJoin(*, X_g[:i1, :i2, :k], Input(θ, :k))))))
    g_log_times = []
    g_log_opt_times = []
    for _ in 1:n_reps
        result_galley = galley(p_g, ST=DCStats, faq_optimizer=optimizer, verbose=0)
        push!(g_log_times, result_galley.execute_time)
        push!(g_log_opt_times, result_galley.opt_time)
    end
    result_log = galley(p_g, ST=DCStats, faq_optimizer=optimizer, verbose=3)

    # ---------------- Covariance Matrix Computation On Many-Many Join -------------------
    cov_g = Query(:P, Materialize(t_undef, t_undef, :j, :k, Σ(:i1, :i2, MapJoin(*, X_g[:i1, :i2, :k], X_g[:i1, :i2, :j]))))
    g_cov_times = []
    g_cov_opt_times = []
    for _ in 1:n_reps
        result_galley = galley(cov_g, ST=DCStats, faq_optimizer=optimizer, verbose=0)
        push!(g_cov_times, result_galley.execute_time)
        push!(g_cov_opt_times, result_galley.opt_time)
    end
    result_cov = galley(cov_g, ST=DCStats, faq_optimizer=optimizer, verbose=3)

    println("Galley Exec: $(mean(g_times[2:end]))")
    println("Galley Opt: $(mean(g_opt_times[2:end]))")
    println("Galley Log Exec: $(mean(g_log_times[2:end]))")
    println("Galley Log Opt: $(mean(g_log_opt_times[2:end]))")
    println("Galley Cov Exec: $(mean(g_cov_times[2:end]))")
    println("Galley Cov Opt: $(mean(g_cov_opt_times[2:end]))")
end

main()
