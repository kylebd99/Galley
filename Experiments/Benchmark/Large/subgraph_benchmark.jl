include("../../Experiments.jl")


#datasets = instances(WORKLOAD)
data = hprd_lite

experiments = ExperimentParams[]
#    push!(experiments, ExperimentParams(workload=data, faq_optimizer=naive))
#push!(experiments, ExperimentParams(workload=data, faq_optimizer=greedy; stats_type=DCStats, warm_start=true, update_cards=false, description="Non-Adaptive", timeout=200))
#push!(experiments, ExperimentParams(workload=data, faq_optimizer=greedy; stats_type=DCStats, warm_start=true, update_cards=true, description="Adaptive", timeout=200))
push!(experiments, ExperimentParams(workload=data, faq_optimizer=greedy; stats_type=DCStats, warm_start=true, description="Ours (greedy)", timeout=200))
push!(experiments, ExperimentParams(workload=data, faq_optimizer=pruned; stats_type=DCStats, warm_start=true, description="Ours (pruned)", timeout=200))
#push!(experiments, ExperimentParams(workload=data, faq_optimizer=greedy; stats_type=DCStats, use_duckdb=true, description="Ours (greedy_cost) + DuckDB", timeout=200))
#push!(experiments, ExperimentParams(workload=data, faq_optimizer=pruned; stats_type=DCStats, use_duckdb=true, description="Ours (pruned) + DuckDB", timeout=200))
#push!(experiments, ExperimentParams(workload=data, faq_optimizer=naive; stats_type=NaiveStats, use_duckdb=true, description="DuckDB", timeout=200))
#push!(experiments, ExperimentParams(workload=data, faq_optimizer=greedy; stats_type=NaiveStats))
#push!(experiments, ExperimentParams(workload=data, faq_optimizer=ordering))

run_experiments(experiments)

graph_grouped_box_plot(experiments; y_type=overall_time, grouping=description, filename="$(data)_subgraph_counting_overall2")
graph_grouped_box_plot(experiments; y_type=opt_time, grouping=description, filename="$(data)_subgraph_counting_opt2")
graph_grouped_box_plot(experiments; y_type=execute_time, grouping=description, filename="$(data)_subgraph_counting_execute2")
graph_grouped_box_plot(experiments; y_type=compile_time, grouping=description, filename="$(data)_subgraph_counting_compile2")
