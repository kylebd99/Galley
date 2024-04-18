using DuckDB

struct DuckDBTensor
    name::String
    columns::Vector{String}
end

function load_to_duckdb(dbconn::DBInterface.Connection, q::PlanNode)
    input_nodes = get_inputs(q)
    for input in input_nodes
        table_name = node_id_to_table_name(input.node_id)
        input.tns = table_name
        tensor = input.tns
        indices = [idx.name for idx in input.idxs]
        if tensor isa Tensor
            fill_table(dbconn, tensor, indices, tensor_name)
        end
    end
end

function tensor_to_vec_of_tuples(t::Tensor)
    return collect(zip(ffindnz(t)...))
end

function create_table(dbconn, idx_names, table_name)
    create_str = "CREATE OR REPLACE TABLE $table_name ("
    prefix = ""
    for idx in idx_names
        create_str *= "$prefix $(idx) INT64"
        prefix = ", "
    end
    create_str *= "$prefix v DOUBLE)"
    DBInterface.execute(dbconn, create_str)
end

function fill_table(dbconn, tensor, idx_names, table_name)
    create_table(dbconn, idx_names, table_name)
    appender = DuckDB.Appender(dbconn, "$table_name")
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

function agg_op_to_duckdb_op(op)
    if op == +
        return "SUM"
    elseif op == min
        return "MIN"
    elseif op == max
        return "MAX"
    end
end

function map_op_to_duckdb_op(op)
    if op == +
        return " + "
    elseif op == *
        return " * "
    elseif op == -
        return " - "
    end
    println("OP NOT RECOGNIZED")
end

node_id_to_table_name(id) = "T_$id"

function delimited_string(args, delimiter)
    output = ""
    prefix = ""
    for arg in args
        output *= prefix * string(arg)
        prefix = delimiter
    end
    return output
end

function get_select_statement(n::PlanNode)
    stmnt = "SELECT "
    agg_op = ""
    output_indices = get_index_set(n.stats)
    if n.kind == Aggregate
        agg_op = agg_op_to_duckdb_op(n.op.val)
        n = n.arg
    end
    all_indices = get_index_set(n.stats)
    if n.kind == MapJoin
        def_val = get_default_value(n.stats)
        map_op = n.op.val
        children = n.args
        child_table_names = Dict(child.node_id => node_id_to_table_name(child.node_id) for child in children)
        v_stmt = "Coalesce($agg_op(" * delimited_string(["Coalesce($tbl.v, 0)" for tbl in values(child_table_names)], map_op_to_duckdb_op(map_op)) * "), $def_val) as v "
        idx_to_child_id = Dict()
        idx_to_table_idx = Dict()
        for idx in all_indices
            for child in children
                if idx ∈ get_index_set(child.stats)
                    idx_to_table_idx[idx] = child_table_names[child.node_id] * ".$idx"
                    idx_to_child_id[idx] = child.node_id
                    break
                end
            end
        end
        attribute_stmt = delimited_string([v_stmt, [idx_to_table_idx[idx] for idx in output_indices]...], ", ")
        stmnt *= attribute_stmt * "\n FROM "
        inner_join = isannihilator(map_op, get_default_value(n.stats))
        is_first = true
        child_join_lines = []
        for child in children
            join_line = "($(get_select_statement(child))) as $(child_table_names[child.node_id])"
            join_equalities = []
            for idx in get_index_set(child.stats)
                if idx_to_child_id[idx] == child.node_id
                    continue
                end
                push!(join_equalities, "$(child_table_names[child.node_id]).$idx=$(idx_to_table_idx[idx])")
            end
            if length(join_equalities) > 0
                join_line *= " ON " * delimited_string(join_equalities, " AND ")
            elseif !is_first
                join_line *= " ON TRUE"
            end
            push!(child_join_lines, join_line)
            is_first = false
        end
        join_delim = inner_join ? "\n INNER JOIN " : "\n OUTER JOIN "
        stmnt *= delimited_string(child_join_lines, join_delim)
        if length(output_indices) > 0
            stmnt *= "\n GROUP BY " * delimited_string([idx_to_table_idx[idx] for idx in output_indices], ", ")
        end
    elseif n.kind == Input
        duckdb_tns = n.tns.val
        tb_idx_to_tns_idx = Dict(n.idxs[i].name => duckdb_tns.columns[i] for i in eachindex(n.idxs))
        v_stmt = "$agg_op(v) as v"
        attribute_stmnt = delimited_string([v_stmt, [tb_idx_to_tns_idx[idx] * " as $idx" for idx in output_indices]...], ", ")
        table_name = duckdb_tns.name
        stmnt *= "$attribute_stmnt \n FROM $table_name"
        if length(output_indices) > 0 && length(output_indices) < length(n.idxs)
            stmnt *= "\n GROUP BY " * delimited_string(output_indices, ", ")
        end
    elseif n.kind == Alias
        v_stmt = "$agg_op(v) as v"
        attribute_stmnt = delimited_string([v_stmt, output_indices...], ", ")
        table_name = alias_to_table_name(n)
        stmnt *= "$attribute_stmnt \n FROM $table_name"
        if length(output_indices) > 0  && length(output_indices) < length(get_index_set(n.stats))
            stmnt *= "\n GROUP BY " * delimited_string(output_indices, ", ")
        end
    end
    return stmnt
end

function get_query_stmnt(q::PlanNode)
    stmnt = "INSERT INTO $(alias_to_table_name(q.name)) BY NAME \n ($(get_select_statement(q.expr)))"
    return stmnt
end

function get_explain_stmnt(q::PlanNode)
    stmnt = "EXPLAIN $(get_select_statement(q.expr))"
    return stmnt
end

function alias_to_table_name(a)
    return replace(string(a.name), "#"=>"")
end

# This function recursively computes the result of bags and produces their output as a table in the
# database under the name "b_$(bag.id)". It then drops the child bag's tables to reduce
# memory consumption.
function _duckdb_compute_query(dbconn, q::PlanNode, verbose)
    table_name = alias_to_table_name(q.name)
    output_indices = get_index_set(q.expr.stats)
    create_table(dbconn, output_indices, table_name)
    explain_stmnt = get_explain_stmnt(q)
    explain_result = @timed DuckDB.execute(dbconn, explain_stmnt)
    opt_time = explain_result.time
    if verbose >=4
        println(explain_result.value)
    end
    query_stmnt = get_query_stmnt(q)
    if verbose >= 3
        println(query_stmnt)
    end
    query_result = @timed DuckDB.execute(dbconn, query_stmnt)
    execute_time = query_result.time
    return opt_time, execute_time
end

function get_inputs(q::PlanNode)
    input_nodes = []
    for n in PostOrderDFS(q)
        if n.kind === Input
            push!(input_nodes, n)
        end
    end
    return input_nodes
end

# First, insert all of the factors as tables e_1, etc, then recursively compute the bags.
# Lastly, return the root bag's table in some format.
function duckdb_execute_query(dbconn, q::PlanNode, verbose)
    insert_time = @elapsed begin
        input_nodes = get_inputs(q)
        for input in input_nodes
            table_name = node_id_to_table_name(input.node_id)
            tensor = input.tns
            indices = [idx.name for idx in input.idxs]
            if tensor isa Tensor
                fill_table(dbconn, tensor, indices, tensor_name)
                input.children[1] = DuckDBTensor(table_name, [string(idx) for idx in indices])
            end
        end
    end
    # If the expression is a materialize, unwrap it
    if q.expr.kind == Materialize
        q = Query(q.name, q.expr.expr)
    end

    # Compute the query and store it in `query.name.name`
    opt_time, execute_time = _duckdb_compute_query(dbconn, q, verbose)
    return (insert_time = insert_time, execute_time  = execute_time, opt_time = opt_time)
end

function _duckdb_query_to_tns(dbconn, query, output_indices)
    output = nothing
    if length(output_indices) > 0
        I = Tuple([Int[] for _ in output_indices])
        V = Float64[]
        M = Tuple(get_dim_size(query.expr.stats, idx) for idx in output_indices)
        for row in DuckDB.execute(dbconn, "SELECT * FROM $(alias_to_table_name(query.name))")
            for i in eachindex(output_indices)
                idx = output_indices[i]
                push!(I[i], row[idx])
            end
            push!(V, row[:v])
        end
        output = Finch.fsparse_impl(I, V, M, +)
    else
        output = only(DuckDB.execute(dbconn, "SELECT * FROM $(alias_to_table_name(query.name))"))[:v]
    end
    return output
end

function _duckdb_drop_alias(dbconn, alias)
    DuckDB.execute(dbconn, "DROP TABLE $(alias_to_table_name(alias))")
end
