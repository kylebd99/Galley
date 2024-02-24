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
