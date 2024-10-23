include("../../Experiments.jl")

#datasets = instances(WORKLOAD)
data = dblp_lite

experiments = ExperimentParams[]

push!(experiments, ExperimentParams(workload=data, faq_optimizer=greedy; stats_type=DCStats, warm_start=true, simple_cse=true, description="Galley", timeout=400))
push!(experiments, ExperimentParams(workload=data, faq_optimizer=greedy; stats_type=DCStats, warm_start=true, simple_cse=false, description="Galley (No CSE)", timeout=400))

run_experiments(experiments)

graph_grouped_box_plot(experiments; y_type=overall_time, grouping=description, filename="$(data)_cse_overall")
graph_grouped_box_plot(experiments; y_type=opt_time, grouping=description, filename="$(data)_cse_opt")
graph_grouped_box_plot(experiments; y_type=execute_time, grouping=description, filename="$(data)_cse_execute")
graph_grouped_box_plot(experiments; y_type=compile_time, grouping=description, filename="$(data)_cse_compile")
