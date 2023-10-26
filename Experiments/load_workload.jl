


function load_workload(workload::WORKLOAD)
    if haskey(IS_SUBGRAPH_WORKLOAD, workload)
        return load_subgraph_workload(workload)
    else
        println("Other Workloads Not Yet Implemented!")
    end
end
