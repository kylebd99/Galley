using DuckDB

# This function recursively computes the result of bags and produces their output as a table in the
# database under the name "b_$(bag.id)". It then drops the child bag's tables to reduce
# memory consumption.
function _duckdb_compute_bag(bag::Bag)

end

# First insert all of the inputs as tables e_1, etc.,
function duckdb_htd_to_output(HTD::HyperTreeDecomposition)

end
