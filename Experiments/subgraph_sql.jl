using LibPQ, Tables
using Galley
using Finch
using IterTools: imap
using DataStructures: counter, inc!

tensor_id_to_table_name(tensor_id, table_name_counters) = "tensor_$(tensor_id)_$(table_name_counters[tensor_id])"

function create_table_umbra(conn, idxs, table_name)
    execute(conn, "DROP TABLE IF EXISTS $table_name")
    create_str = "CREATE TABLE $table_name ("
    prefix = ""
    for idx in idxs
        create_str *= "$prefix $(idx) INT"
        prefix = ", "
    end
    create_str *= "$prefix v INT)"
    create_str = replace(create_str, "#"=>"")
    execute(conn, create_str)
end

function tensor_to_table(t::Tensor, idxs)
    return  zip(ffindnz(t)...)
end

#= 
function fill_table_umbra(conn, tensor, idxs, table_name)
    create_table_umbra(conn, idxs, table_name)
    tensor_data = tensor_to_table(tensor, idxs)
    
    execute(conn, "BEGIN;")
    insert_statement = "INSERT INTO $table_name ("
    for idx in idxs 
        insert_statement *= "$idx, "
    end

    insert_statement *= " v) VALUES ("
    prefix = ""
    for i in range(1, length(idxs) + 1)
        insert_statement *= "$prefix\$$i"
        prefix = ", "
    end
    insert_statement *= ");"
    LibPQ.load!(
        tensor_data,
        conn,
        insert_statement
    )
    execute(conn, "COMMIT;")
end
=#

function fill_table_umbra(conn, tensor, idxs, table_name)
    create_table_umbra(conn, idxs, table_name)
    tensor_data = tensor_to_table(tensor, idxs)
    
    row_strings = imap(tensor_data) do idx_and_vals
        row_string = ""
        prefix = ""
        for val in idx_and_vals
            row_string *= "$prefix$val"
            prefix = ","
        end
        row_string *= "\n"
        return row_string
    end
    
    copyin = LibPQ.CopyIn("COPY $table_name FROM STDIN (FORMAT CSV);", row_strings)
    execute(conn, copyin)
end

function get_sql_query(table_names, table_idxs)
    table_aliases = ["A_$i" for i in eachindex(table_names)]
    sql_query = "SELECT SUM("
    prefix = ""
    for name in table_aliases
        sql_query *= "$prefix $name.v"
        prefix = " *"
    end

    idx_to_root_alias = Dict()
    for i in eachindex(table_names)
        table_name = table_names[i]
        for idx in table_idxs[table_name]
            if !haskey(idx_to_root_alias, idx)
                idx_to_root_alias[idx] = table_aliases[i]
            end
        end
    end

    sql_query *= ") \nFROM $(table_names[1]) as $(table_aliases[1])\n"
    for i in 2:length(table_names)
        prefix = ""
        table = table_names[i]
        alias = table_aliases[i]
        sql_query *= "INNER JOIN $(table) as $(alias) ON ("
        has_clause = false
        for idx in table_idxs[table]
            if alias != idx_to_root_alias[idx]
                sql_query *= "$prefix $alias.$idx = $(idx_to_root_alias[idx]).$idx"
                prefix = " AND"
                has_clause = true
            end
        end
        if !has_clause
            sql_query *= "true"
        end
        sql_query *= ")\n"
    end
    sql_query *= ";"
    return sql_query    
end

# The query needs to take the form of:
# Query(:out, Materialize(Aggregate(+, MapJoin(*, Input(TensorId, idxs...)...))))
# i.e. no tree of mapjoin operators
function execute_galley_query_umbra(query::PlanNode)
    conn = LibPQ.Connection("host=127.0.0.1 port=5432 user=postgres password=postgres")
    aggregate = query.expr.expr
    mapjoin = aggregate.arg
    inputs = mapjoin.args
    table_names = []
    table_name_counters = counter(String)
    table_idxs = Dict()
    insert_start = time()
    for input in inputs
        @assert input.kind == Input
        idxs = [idx.name for idx in input.idxs]
        inc!(table_name_counters, input.id)
        table_name = tensor_id_to_table_name(input.id, table_name_counters)
        fill_table_umbra(conn, input.tns.val, idxs, table_name)
        push!(table_names, table_name)
        table_idxs[table_name] = idxs
    end
    println("Insert Time: $(time() - insert_start)")

    sql_query = get_sql_query(table_names, table_idxs)
    println(sql_query)

    execute_start = time() 
    result = columntable(execute(conn, sql_query))
    execute_time = time()-execute_start
    return (result = result.sum, execute_time = execute_time)
end
