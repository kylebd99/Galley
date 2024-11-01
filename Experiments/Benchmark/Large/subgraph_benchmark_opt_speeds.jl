include("../../Experiments.jl")

data = aids

experiments = ExperimentParams[]
push!(experiments, ExperimentParams(workload=data, faq_optimizer=greedy; stats_type=DCStats, warm_start=true, description="Greedy", timeout=600))
push!(experiments, ExperimentParams(workload=data, faq_optimizer=pruned; stats_type=DCStats, warm_start=true, description="Branch-and-Bound", timeout=600))
push!(experiments, ExperimentParams(workload=data, faq_optimizer=exact; stats_type=DCStats, warm_start=true, description="Exact", timeout=600))

#run_experiments(experiments)

graph_grouped_bar_plot(experiments; y_type=opt_time, grouping=description, y_label="Mean Optimization Time", group_order=["Branch-and-Bound", "Greedy", "Exact"], filename="$(data)_opt_speeds_opt")
