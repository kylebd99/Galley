include("../../Experiments.jl")


#datasets = instances(WORKLOAD)
data = aids

experiments = ExperimentParams[]
push!(experiments, ExperimentParams(workload=data, faq_optimizer=greedy; stats_type=DCStats, warm_start=true, max_kernel_size=2, description="Max Size 2", timeout=200))
push!(experiments, ExperimentParams(workload=data, faq_optimizer=greedy; stats_type=DCStats, warm_start=true, max_kernel_size=3, description="Max Size 3", timeout=200))
push!(experiments, ExperimentParams(workload=data, faq_optimizer=greedy; stats_type=DCStats, warm_start=true, max_kernel_size=4, description="Max Size 4", timeout=200))
push!(experiments, ExperimentParams(workload=data, faq_optimizer=greedy; stats_type=DCStats, warm_start=true, max_kernel_size=5, description="Max Size 5", timeout=200))
push!(experiments, ExperimentParams(workload=data, faq_optimizer=greedy; stats_type=DCStats, warm_start=true, max_kernel_size=6, description="Max Size 6", timeout=200))
push!(experiments, ExperimentParams(workload=data, faq_optimizer=greedy; stats_type=DCStats, warm_start=true, max_kernel_size=7, description="Max Size 7", timeout=200))

run_experiments(experiments)

graph_grouped_box_plot(experiments; y_type=overall_time, grouping=description, filename="$(data)_kernel_sizes_overall")
graph_grouped_box_plot(experiments; y_type=opt_time, grouping=description, filename="$(data)_kernel_sizes_opt")
graph_grouped_box_plot(experiments; y_type=execute_time, grouping=description, filename="$(data)_kernel_sizes_execute")
graph_grouped_box_plot(experiments; y_type=compile_time, grouping=description, filename="$(data)_kernel_sizes_compile")
