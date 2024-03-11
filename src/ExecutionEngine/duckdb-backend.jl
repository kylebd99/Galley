using DuckDB

struct DuckDBTensor
    name::String
    columns::Vector{String}
end

function load_to_duckdb(dbconn::DBInterface.Connection ,faq::FAQInstance)
    for factor in faq.factors
        tensor_name = factor_to_table_name(factor)
        tensor = factor.input.args[2]
        indices = factor.input.args[1]
        fill_table(dbconn, tensor, indices, tensor_name)
        factor.input.args[2] = DuckDBTensor(tensor_name, [idx.name for idx in indices])
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
    opt_time = 0
    for b in bag.child_bags
        opt_time += _duckdb_compute_bag(dbconn, b)
    end

    bag_table = bag_to_table_name(bag)
    create_table(dbconn, bag.parent_indices, bag_table)

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

    table_names_and_aliases = ∪([bag_to_table_name(b) for b in bag.child_bags],
                                ["$(f.input.args[2].name) as $(factor_to_table_name(f))" for f in bag.edge_covers])

    table_aliases = ∪([bag_to_table_name(b) for b in bag.child_bags],
                        [factor_to_table_name(f) for f in bag.edge_covers])

    canonical_parent_refs = []
    for idx in bag.parent_indices
        finished = false
        for child_bag in bag.child_bags
            if idx in child_bag.parent_indices
                push!(canonical_parent_refs, "$(bag_to_table_name(child_bag)).$idx")
                finished = true
                break
            end
        end
        for f in bag.edge_covers
            finished && break
            if idx in f.all_indices
                table_indices = f.input.args[2].columns
                idx_pos = findfirst(x->x==idx, f.input.args[1])
                push!(canonical_parent_refs, "$(factor_to_table_name(f)).$(table_indices[idx_pos])")
                break
            end
        end
    end

    query_str = "INSERT INTO $bag_table \nSELECT "
    prefix = ""
    for ref in canonical_parent_refs
        query_str *= "$prefix$ref"
        prefix = ", "
    end
    query_str *= "$(prefix)SUM("
    prefix = ""
    for tbl in table_aliases
        query_str *= "$prefix$tbl.v"
        prefix = " * "
    end
    query_str *= ") as v"


    query_str *= "\nFROM "
    prefix = ""
    for tbl in table_names_and_aliases
        query_str *= "$prefix$tbl"
        prefix = ", "
    end

    prefix = "\nWHERE "
    for idx in bag.covered_indices
        idx_refs = []
        for f in bag.edge_covers
            if idx in f.all_indices
                table_indices = f.input.args[2].columns
                idx_pos =findfirst(x->x==idx, f.input.args[1])
                push!(idx_refs, "$(factor_to_table_name(f)).$(table_indices[idx_pos])")
            end
        end
        for b in bag.child_bags
            if idx in b.parent_indices
                push!(idx_refs, "$(bag_to_table_name(b)).$idx")
            end
        end

        if length(idx_refs) == 1
            continue
        end
        base_ref = idx_refs[1]
        for ref in idx_refs[2:end]
            query_str *= "$prefix$base_ref = $ref"
            prefix = " AND "
        end
    end

    if length(bag.parent_indices) > 0
        query_str *= "\nGROUP BY "
        prefix = ""
        for ref in canonical_parent_refs
            query_str *= "$prefix$ref"
            prefix = ", "
        end
    end

    explain_str = "Explain $query_str"
    opt_time += @elapsed DuckDB.execute(dbconn, explain_str)


    DuckDB.execute(dbconn, query_str)

    for tbl in bag.child_bags
        DuckDB.execute(dbconn, "DROP TABLE $(bag_to_table_name(tbl))")
    end
    return opt_time
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
    result = @timed _duckdb_compute_bag(dbconn, htd.root_bag)
    opt_time = result.value
    execute_time = result.time - opt_time


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

    DuckDB.execute(dbconn, "DROP TABLE $(bag_to_table_name(htd.root_bag))")
    return (value = output, insert_time = insert_time, execute_time  = execute_time, opt_time = opt_time)
end
