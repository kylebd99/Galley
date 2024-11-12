using DuckDB

struct DuckDBTensor
    name::String
    columns::Vector{String}
end

function Base.size(x::DuckDBTensor)
    return length(x.columns)
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

function create_table(dbconn, idx_names, table_name, val_type)
    create_str = "CREATE OR REPLACE TABLE $table_name ("
    prefix = ""
    for idx in idx_names
        create_str *= "$prefix $(idx) INT64"
        prefix = ", "
    end
    create_str *= "$prefix v $(type_to_duckdb_type(val_type)))"
    create_str = replace(create_str, "#"=>"")
    DBInterface.execute(dbconn, create_str)
end

function fill_table(dbconn, tensor, idx_names, table_name)
    create_table(dbconn, idx_names, table_name, typeof(default(tensor)))
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
    elseif op isa Finch.Chooser
        return "ANY_VALUE"
    end
end


function map_op_to_infix_duckdb_op(op)
    if op == +
        return " + "
    elseif op == *
        return " * "
    elseif op == -
        return " - "
    elseif op == |
        return " OR "
    elseif op == &
        return " AND "
    end
    println("OP NOT RECOGNIZED")
end

function type_to_duckdb_type(t)
    if t == Float64
        return "DOUBLE"
    elseif t == Int64
        return "INT64"
    elseif t == Bool
        return "BOOL"
    end
end

function map_op_to_prefix_duckdb_op(op)
    if op == max
        return "greatest"
    elseif op == min
        return "least"
    elseif op == !
        return "NOT"
    end
    println("OP NOT RECOGNIZED")
end

is_infix_op(op) = op in (+, *, -, |, &)

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

function get_select_statement(n::PlanNode, indent_count = 0)
    stmnt = "SELECT "
    indents = prod([" " for _ in 1:(2*indent_count)]; init = "")
    agg_op = ""
    output_indices = get_index_set(n.stats)
    if n.kind == Aggregate
        agg_op = agg_op_to_duckdb_op(n.op.val)
        n = n.arg
    end
    all_indices = get_index_set(n.stats)
    if n.kind == MapJoin
        def_val = get_default_value(n.stats)
        if agg_op == ""
            agg_op = "ANY_VALUE"
        end
        map_op = n.op.val
        children = n.args
        child_defaults = [get_default_value(child.stats) for child in children if child.kind in (Alias, Input, MapJoin)]
        child_tables = [child for child in children if child.kind in (Alias, Input, MapJoin)]
        child_table_names = Dict(child.node_id => node_id_to_table_name(child.node_id) for child in child_tables)
        literals = ["CAST($(c.val) as $(type_to_duckdb_type(typeof(c.val))))" for c in children if c.kind === Value]
        v_stmt_inputs = [["Coalesce($(child_table_names[child_tables[i].node_id]).v, $(child_defaults[i]))" for i in eachindex(child_tables)]..., literals...]
        v_stmt = "Coalesce($agg_op("
        if is_infix_op(map_op)
            v_stmt *= delimited_string(v_stmt_inputs, map_op_to_infix_duckdb_op(map_op))
        else
            v_stmt *= map_op_to_prefix_duckdb_op(map_op) * "(" * delimited_string(v_stmt_inputs, ",") * ")"
        end
        v_stmt *= "), $def_val) as v "
        idx_to_child_id = Dict()
        idx_to_output_idx = Dict()
        idx_to_coalesce_idx = Dict()
        idx_to_table_idx = Dict()
        for idx in all_indices
            relevant_children = []
            for child in child_tables
                if idx ∈ get_index_set(child.stats)
                    push!(relevant_children, child_table_names[child.node_id] * ".$idx")
                    if !haskey(idx_to_table_idx, idx)
                        idx_to_table_idx[idx] = child_table_names[child.node_id] * ".$idx"
                        idx_to_child_id[idx] = child.node_id
                    end
                end
            end
            idx_to_output_idx[idx] = "Coalesce($(delimited_string(relevant_children, ","))) as $idx"
            idx_to_coalesce_idx[idx] = "Coalesce($(delimited_string(relevant_children, ",")))"
        end
        attribute_stmt = delimited_string([v_stmt, [idx_to_output_idx[idx] for idx in output_indices]...], ", ")
        stmnt *= attribute_stmt * "\n$(indents)FROM "
        inner_join = isannihilator(map_op, get_default_value(n.stats))
        is_first = true
        child_join_lines = []
        for child in child_tables
            join_line = "($(get_select_statement(child, indent_count + 1))) as $(child_table_names[child.node_id])"
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
        join_delim = inner_join ? "\n$(indents)INNER JOIN " : "\n$(indents)FULL OUTER JOIN "
        stmnt *= delimited_string(child_join_lines, join_delim)
        if length(output_indices) > 0
            stmnt *= "\n$(indents)GROUP BY " * delimited_string([idx_to_coalesce_idx[idx] for idx in output_indices], ", ")
        end
    elseif n.kind == Input
        has_groupby = length(output_indices) > 0 && length(output_indices) < length(n.idxs)
        if has_groupby && agg_op == ""
            agg_op = "ANY_VALUE"
        end
        duckdb_tns = n.tns.val
        tb_idx_to_tns_idx = Dict(n.idxs[i].name => duckdb_tns.columns[i] for i in eachindex(n.idxs))
        v_stmt = has_groupby ? "$agg_op(v) as v" : " v"
        attribute_stmnt = delimited_string([v_stmt, [tb_idx_to_tns_idx[idx] * " as $idx" for idx in output_indices]...], ", ")
        table_name = duckdb_tns.name
        stmnt *= "$attribute_stmnt \n$(indents)FROM $table_name"
        if has_groupby
            stmnt *= "\n$(indents)GROUP BY " * delimited_string(output_indices, ", ")
        end
    elseif n.kind == Alias
        has_groupby = length(output_indices) > 0 && length(output_indices) < length(get_index_set(n.stats))
        if has_groupby && agg_op == ""
            agg_op = "ANY_VALUE"
        end

        v_stmt = has_groupby ? "$agg_op(v) as v" : " v"
        attribute_stmnt = delimited_string([v_stmt, output_indices...], ", ")
        table_name = alias_to_table_name(n)
        stmnt *= "$attribute_stmnt \n$(indents)FROM $table_name"
        if has_groupby
            stmnt *= "\n$(indents)GROUP BY " * delimited_string(output_indices, ", ")
        end
    end
    return stmnt
end

function get_query_stmnt(q::PlanNode)
    stmnt = "INSERT INTO $(alias_to_table_name(q.name)) BY NAME \n ($(get_select_statement(q.expr)))"
    stmnt = replace(stmnt, "#"=>"")
    return stmnt
end

function get_explain_stmnt(q::PlanNode)
    stmnt = "EXPLAIN $(get_select_statement(q.expr))"
    stmnt = replace(stmnt, "#"=>"")
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
    output_type = typeof(get_default_value(q.expr.stats))
    create_table(dbconn, output_indices, table_name, output_type)
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


# First, insert all of the factors as tables e_1, etc, then recursively compute the bags.
# Lastly, return the root bag's table in some format.
function duckdb_execute_query(dbconn, q::PlanNode, verbose)
    insert_time = @elapsed begin
        input_nodes = get_inputs(q)
        for input in input_nodes
            table_name = node_id_to_table_name(input.node_id)
            tensor = input.tns.val
            indices = [idx.name for idx in input.idxs]
            if tensor isa Tensor
                fill_table(dbconn, tensor, indices, table_name)
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
        V = []
        M = Tuple(get_dim_size(query.expr.stats, idx.name) for idx in output_indices)
        for row in DuckDB.execute(dbconn, "SELECT * FROM $(alias_to_table_name(query.name))")
            for i in eachindex(output_indices)
                idx = output_indices[i].name
                ismissing(row[idx]) && continue
                push!(I[i], row[idx])
            end
            push!(V, row[:v])
        end
        V = Vector{typeof(V[1])}(V)
        output = Finch.fsparse_impl(I, V, M, +)
    else
        output = only(DuckDB.execute(dbconn, "SELECT * FROM $(alias_to_table_name(query.name))"))[:v]
    end
    return output
end

function _duckdb_drop_alias(dbconn, alias)
    DuckDB.execute(dbconn, "DROP TABLE $(alias_to_table_name(alias))")
end


function duckdb_execute_logical_plan(logical_queries, dbconn, output_name, output_order, faq_time, verbose)
    duckdb_opt_time = faq_time
    duckdb_exec_time = 0
    duckdb_insert_time = 0
    for query in logical_queries
        verbose >= 1 && println("-------------- Computing Alias $(query.name) -------------")
        query_timings = duckdb_execute_query(dbconn, query, verbose)
        verbose >= 1 && println("$query_timings")
        duckdb_opt_time += query_timings.opt_time
        duckdb_exec_time += query_timings.execute_time
        duckdb_insert_time += query_timings.insert_time
    end
    result = _duckdb_query_to_tns(dbconn, logical_queries[end], output_order)
    for query in logical_queries
        _duckdb_drop_alias(dbconn, query.name)
    end
    verbose >= 1 && println("Time to Optimize: ",  duckdb_opt_time)
    verbose >= 1 && println("Time to Insert: ", duckdb_insert_time)
    verbose >= 1 && println("Time to Execute: ", duckdb_exec_time)
    return (value=[result],
                opt_time=duckdb_opt_time,
                insert_time = duckdb_insert_time,
                execute_time=duckdb_exec_time,
                overall_time = duckdb_opt_time + duckdb_insert_time + duckdb_exec_time)

end
