using DuckDB
using Galley
using Galley: faq_to_htd, duckdb_htd_to_output
using Finch

function duckdb_compute_faq(faq::FAQInstance)
    dbconn = DBInterface.connect(DuckDB.DB, ":memory:")
    db_faq = deepcopy(faq)
    load_to_duckdb(dbconn, db_faq)
    db_htd = faq_to_htd(db_faq)
    result = duckdb_htd_to_output(dbconn, db_htd)
    return (time = result.execute_time, result=result.value)
end
