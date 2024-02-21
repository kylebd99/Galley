

@enum WORKLOAD  aids human lubm80 yago yeast hprd wordnet dblp youtube eu2005 patents


const IS_GCARE_DATASET = Dict(aids=>true, human=>true, lubm80=>true, yago=>true,
     yeast=>false, hprd=>false, wordnet=>false, dblp=>false, youtube=>false, eu2005=>false, patents=>false)

const IS_SUBGRAPH_WORKLOAD = Dict(aids=>true, human=>true, lubm80=>true, yago=>true,
     yeast=>true, hprd=>true, wordnet=>true, dblp=>true, youtube=>true, eu2005=>true, patents=>true)


struct ExperimentParams
    workload::WORKLOAD
    warm_start::Bool
    faq_optimizer::FAQ_OPTIMIZERS
    stats_type::Type
    use_duckdb::Bool
    description::String

    function ExperimentParams(;workload=human, warm_start=false, faq_optimizer=naive, stats_type = DCStats, use_duckdb=false, description="")
        return new(workload, warm_start, faq_optimizer, stats_type, use_duckdb, description)
    end
end

function param_to_results_filename(param::ExperimentParams)
    if param.use_duckdb
        return "DuckDB.csv"
    end
    filename = ""
    filename *= string(param.workload) * "_"
    filename *= string(param.warm_start) * "_"
    filename *= string(param.faq_optimizer) * "_"
    filename *= string(param.stats_type) * ".csv"
    return filename
end
