include("../../Experiments.jl")


#datasets = instances(WORKLOAD)
data = dblp_lite

experiments = ExperimentParams[]

push!(experiments, ExperimentParams(workload=data, faq_optimizer=greedy; stats_type=DCStats, warm_start=true, update_cards=true, description="Ours (Adaptive)", timeout=400))
push!(experiments, ExperimentParams(workload=data, faq_optimizer=greedy; stats_type=DCStats, warm_start=true, update_cards=false, description="Ours (Non-Adaptive)", timeout=400))

run_experiments(experiments)

graph_grouped_box_plot(experiments; y_type=overall_time, grouping=description, filename="$(data)_adaptive_overall")
graph_grouped_box_plot(experiments; y_type=opt_time, grouping=description, filename="$(data)_adaptive_opt")
graph_grouped_box_plot(experiments; y_type=execute_time, grouping=description, filename="$(data)_adaptive_execute")
graph_grouped_box_plot(experiments; y_type=compile_time, grouping=description, filename="$(data)_adaptive_compile")
