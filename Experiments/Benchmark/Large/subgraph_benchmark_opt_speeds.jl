include("../../Experiments.jl")

data = aids

experiments = ExperimentParams[]
push!(experiments, ExperimentParams(workload=data, faq_optimizer=greedy; stats_type=DCStats, warm_start=true, description="Greedy", timeout=400))
push!(experiments, ExperimentParams(workload=data, faq_optimizer=pruned; stats_type=DCStats, warm_start=true, description="Pruned", timeout=400))
push!(experiments, ExperimentParams(workload=data, faq_optimizer=exact; stats_type=DCStats, warm_start=true, description="Exact", timeout=400))

run_experiments(experiments)

graph_grouped_box_plot(experiments; y_type=overall_time, grouping=description, filename="$(data)_opt_speeds_overall")
graph_grouped_box_plot(experiments; y_type=opt_time, grouping=description, filename="$(data)_opt_speeds_opt")
graph_grouped_box_plot(experiments; y_type=execute_time, grouping=description, filename="$(data)_opt_speeds_execute")
graph_grouped_box_plot(experiments; y_type=compile_time, grouping=description, filename="$(data)_opt_speeds_compile")
