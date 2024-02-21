using DuckDB
using Galley
using Finch

function tensor_to_vec_of_tuples(t::Tensor)
    s = size(t)
    coo_tensor = Tensor(SparseCOOLevel(Element(Finch.default(t)), s))
    copyto!(coo_tensor, t)
    index_tuples = coo_tensor.lvl.tbl
    index_vector = zip(index_tuples...)
    DT = typeof(Finch.default(t))
    values_vector = DT[]
    for indices in index_vector
        push!(values_vector, coo_tensor[indices...])
    end
    return collect(zip(index_tuples..., values_vector))
end

function create_table(conn, tensor, idx_names, table_name)
    create_str = "CREATE TABLE $table_name ("
    prefix = ""
    for idx in idx_names
        create_str *= "$prefix $(idx.name) INTEGER"
        prefix = ", "
    end
    create_str *= ", v FLOAT)"
    DBInterface.execute(conn, create_str)
    appender = DuckDB.Appender(conn, "$table_name")
    data = tensor_to_vec_of_tuples(tensor)
    for row in data
        for val in row
            DuckDB.append(appender, val)
        end
        DuckDB.end_row(appender)
    end
    DuckDB.flush(appender)
    DuckDB.close(appender)
end


function duckdb_compute_faq(faq::FAQInstance)
    conn = DBInterface.connect(DuckDB.DB, ":memory:")
    factors = collect(faq.factors)
    factor_names = ["T_$i" for i in eachindex(factors)]
    for i in eachindex(factors)
        factor = factors[i]
        tensor_name = factor_names[i]
        tensor = factor.input.args[2]
        indices = factor.input.args[1]
        create_table(conn, tensor, indices, tensor_name)
    end

    query_str = "SELECT SUM("
    prefix = ""
    for i in eachindex(factors)
        query_str *= "$prefix$(factor_names[i]).v"
        prefix = " * "
    end
    query_str *= ") as output\nFROM "
    prefix = ""
    for i in eachindex(factors)
        query_str *= "$prefix$(factor_names[i])"
        prefix = ", "
    end
    query_str *= "\nWHERE "
    prefix = ""
    for idx in faq.input_indices
        idx_factors = []
        for i in eachindex(factors)
            if idx in factors[i].all_indices
                push!(idx_factors, factor_names[i])
            end
        end
        if length(idx_factors) == 1
            continue
        end
        base_factor = idx_factors[1]
        for f in idx_factors[2:end]
            query_str *= "$prefix$base_factor.$idx = $f.$idx"
            prefix = " AND "
        end
    end
    time = @elapsed DuckDB.execute(conn, query_str)
    result = only(DuckDB.execute(conn, query_str))[:output]
    return (time = time, result=result)
end
