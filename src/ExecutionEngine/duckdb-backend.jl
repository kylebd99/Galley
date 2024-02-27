using DuckDB

function load_to_duckdb(dbconn::DBInterface.Connection ,faq::FAQInstance)
    for factor in faq.factors
        tensor_name = factor_to_table_name(factor)
        tensor = factor.input.args[2]
        indices = factor.input.args[1]
        fill_table(dbconn, tensor, indices, tensor_name)
        factor.input.args[2] = tensor_name
    end
end


function tensor_to_vec_of_tuples(t::Tensor)
    return collect(zip(ffindnz(t)...))
end

function create_table(dbconn, idx_names, table_name)
    create_str = "CREATE OR REPLACE TABLE $table_name ("
    prefix = ""
    for idx in idx_names
        create_str *= "$prefix $(idx.name) INT64"
        prefix = ", "
    end
    create_str *= "$prefix v DOUBLE)"
    DBInterface.execute(dbconn, create_str)
end

function fill_table(dbconn, tensor, idx_names, table_name)
    create_table(dbconn, idx_names, table_name)
    appender = DuckDB.Appender(dbconn, "$table_name")
    data = tensor_to_vec_of_tuples(tensor)
#    println("$table_name Length: $(length(data)), countstored: $(countstored(tensor)), indices: $(idx_names)")

    for row in data
        for val in row
            DuckDB.append(appender, val)
        end
        DuckDB.end_row(appender)
    end
    DuckDB.flush(appender)
    DuckDB.close(appender)
end

bag_to_table_name(bag::Bag) = "b_$(bag.id)"
factor_to_table_name(factor::Factor) = "f_$(factor.id)"

# This function recursively computes the result of bags and produces their output as a table in the
# database under the name "b_$(bag.id)". It then drops the child bag's tables to reduce
# memory consumption.
function _duckdb_compute_bag(dbconn, bag::Bag)

    for b in bag.child_bags
        _duckdb_compute_bag(dbconn, b)
    end

    bag_table = bag_to_table_name(bag)
    create_table(dbconn, bag.parent_indices, bag_table)
    child_tables = ∪([bag_to_table_name(b) for b in bag.child_bags], [factor_to_table_name(f) for f in bag.edge_covers])
    var_to_covered_var = Dict()
    for var in bag.covered_indices
        for b in bag.child_bags
            if var in b.parent_indices
                var_to_covered_var[var] = "$(bag_to_table_name(b)).$var"
                break
            end
        end
        for f in bag.edge_covers
            if var in f.all_indices
                var_to_covered_var[var] = "$(factor_to_table_name(f)).$var"
                break
            end
        end
    end


    query_str = "INSERT INTO $bag_table \nSELECT "
    prefix = ""
    for idx in bag.parent_indices
        query_str *= "$prefix$(var_to_covered_var[idx])"
        prefix = ", "
    end
    query_str *= "$(prefix)SUM("
    prefix = ""
    for tbl in child_tables
        query_str *= "$prefix$tbl.v"
        prefix = " * "
    end
    query_str *= ") as v"


    query_str *= "\nFROM "

    prefix = ""
    for tbl in child_tables
        query_str *= "$prefix$tbl"
        prefix = ", "
    end

    prefix = "\nWHERE "
    for idx in bag.covered_indices
        idx_tables = []
        for f in bag.edge_covers
            if idx in f.all_indices
                push!(idx_tables, factor_to_table_name(f))
            end
        end
        for b in bag.child_bags
            if idx in b.parent_indices
                push!(idx_tables, bag_to_table_name(b))
            end
        end

        if length(idx_tables) == 1
            continue
        end
        base_factor = idx_tables[1]
        for f in idx_tables[2:end]
            query_str *= "$prefix$base_factor.$idx = $f.$idx"
            prefix = " AND "
        end
    end

    if length(bag.parent_indices) > 0
        query_str *= "\nGROUP BY "
        prefix = ""
        for idx in bag.parent_indices
            query_str *= "$prefix$(var_to_covered_var[idx])"
            prefix = ", "
        end
    end

    DuckDB.execute(dbconn, query_str)

    for tbl in child_tables
        DuckDB.execute(dbconn, "DROP TABLE $tbl")
    end
end

function _collect_factors(bag::Bag)
    factor_list = Factor[]
    append!(factor_list, bag.edge_covers)
    for b in bag.child_bags
        append!(factor_list, _collect_factors(b))
    end
    return factor_list
end

function fsparse_fixed(I, V, M, combine)
    C = map(tuple, reverse(I)...)
    updater = false
    if !issorted(C)
        P = sortperm(C)
        C = C[P]
        V = V[P]
        updater = true
    end
    if !allunique(C)
        P = unique(p -> C[p], 1:length(C))
        C = C[P]
        push!(P, length(I[1]) + 1)
        V = map((start, stop) -> foldl(combine, @view V[start:stop - 1]), P[1:end - 1], P[2:end])
        updater = true
    end
    if updater
        I = map(i -> similar(i, length(C)), I)
        foreach(((p, c),) -> ntuple(n->I[n][p] = c[n], length(I)), enumerate(C))
        I = reverse(I)
    else
        I = map(copy, I)
    end
    return fsparse!(I..., V, M)
end

# First, insert all of the factors as tables e_1, etc, then recursively compute the bags.
# Lastly, return the root bag's table in some format.
function duckdb_htd_to_output(dbconn, htd::HyperTreeDecomposition)
    insert_time = @elapsed begin
        factors = _collect_factors(htd.root_bag)
        for factor in factors
            tensor_name = factor_to_table_name(factor)
            tensor = factor.input.args[2]
            indices = factor.input.args[1]
            if tensor isa Tensor
                fill_table(dbconn, tensor, indices, tensor_name)
            end
        end
    end
    execute_time = @elapsed _duckdb_compute_bag(dbconn, htd.root_bag)

    output = nothing
    if !isnothing(htd.output_index_order) && length(htd.output_index_order) > 0
        output_index_order = htd.output_index_order
        I = Tuple([Int[] for _ in htd.output_index_order])
        V = Float64[]
        M = Tuple(get_dim_size(htd.root_bag.stats, idx) for idx in output_index_order)
        for row in DuckDB.execute(dbconn, "SELECT * FROM $(bag_to_table_name(htd.root_bag))")
            for i in eachindex(output_index_order)
                idx = output_index_order[i]
                push!(I[i], row[Symbol(idx.name)])
            end
            push!(V, row[:v])
        end
        output = fsparse_fixed(I, V, M, +)
    else
        output = only(DuckDB.execute(dbconn, "SELECT * FROM $(bag_to_table_name(htd.root_bag))"))[:v]
    end
    return (value = output, insert_time = insert_time, execute_time  = execute_time)
end
