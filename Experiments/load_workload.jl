


function load_workload(workload::WORKLOAD, stats_type::Type, dbconn)
    if haskey(IS_SUBGRAPH_WORKLOAD, workload)
        return load_subgraph_workload(workload, stats_type, dbconn)
    else
        println("Other Workloads Not Yet Implemented!")
    end
end
