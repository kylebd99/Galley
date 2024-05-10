using Galley
using Finch
using BenchmarkTools
using Galley: initmax, _calc_dc_from_structure, IndexExpr

include("../../Experiments.jl")

function query_triangle(e1, e2, e3)
    e1 = relabel_input(e1, :i, :j)
    e2 = relabel_input(e2, :k, :j)
    e3 = relabel_input(e3, :i, :k)
    query = Query(:out, Materialize(Aggregate(+, :i, :j, :k, MapJoin(*, e1, e2, e3))))
    return query
end


function query_path(e1, e2, e3, e4)
    e1 = relabel_input(e1, :i, :j)
    e2 = relabel_input(e2, :j, :k)
    e3 = relabel_input(e3, :k, :l)
    e4 = relabel_input(e4, :l, :m)
    query = Query(:out, Materialize(Aggregate(+, :i, :j, :k, :l, :m, MapJoin(*, e1, e2, e3, e4))))
    return query
end

function query_bowtie(e1, e2, e3, e4, e5, e6)
    e1 = relabel_input(e1, :i, :j)
    e2 = relabel_input(e2, :j, :k)
    e3 = relabel_input(e3, :i, :k)
    e4 = relabel_input(e4, :l, :m)
    e5 = relabel_input(e5, :l, :k)
    e6 = relabel_input(e6, :m, :k)
    query = Query(:out, Materialize(Aggregate(+, :i, :j, :k, :l, :m, MapJoin(*, e1, e2, e3, e4, e5, e6))))
    return query
end

time_dict = Dict("balanced triangle"=>Dict(),
                "unbalanced triangle"=>Dict(),
                "balanced path"=>Dict(),
                "unbalanced path"=>Dict(),
                "balanced bowtie"=>Dict(),
                "unbalanced bowtie"=>Dict(), )

verbosity=1

for ST in [DCStats, NaiveStats]
    vertices, edges = load_dataset("Experiments/Data/Subgraph_Data/yeast/yeast.graph", ST, nothing,subgraph_matching_data=true)
    main_edge = edges[0]

    qt_balanced = query_triangle(main_edge, main_edge, main_edge)
    qt_balanced_time = galley(qt_balanced, faq_optimizer=greedy, ST=ST, verbose=verbosity)
    qt_balanced_time = galley(qt_balanced, faq_optimizer=greedy, ST=ST, verbose=verbosity)
    println("Balanced Triangle [$ST]: ", qt_balanced_time)
    time_dict["balanced triangle"][string(ST)] = qt_balanced_time

    qp_balanced = query_path(main_edge, main_edge, main_edge, main_edge)
    qp_balanced_time = galley(qp_balanced, faq_optimizer=greedy, ST=ST, verbose=verbosity)
    qp_balanced_time = galley(qp_balanced, faq_optimizer=greedy, ST=ST, verbose=verbosity)
    println("Balanced Path [$ST]: ", qp_balanced_time)
    time_dict["balanced path"][string(ST)] = qp_balanced_time

    qb_balanced = query_bowtie(main_edge, main_edge, main_edge, main_edge, main_edge, main_edge)
    qb_balanced_time = galley(qb_balanced, faq_optimizer=greedy, ST=ST, verbose=verbosity)
    qb_balanced_time = galley(qb_balanced, faq_optimizer=greedy, ST=ST, verbose=verbosity)
    println("Balanced Bowtie [$ST]: ", qb_balanced_time)
    time_dict["balanced bowtie"][string(ST)] = qb_balanced_time
end

for ST in [DCStats, NaiveStats]
    dbconn = DBInterface.connect(DuckDB.DB, ":memory:")
    vertices, edges = load_dataset("Experiments/Data/Subgraph_Data/yeast/yeast.graph", ST, dbconn,subgraph_matching_data=true)
    main_edge = edges[0]

    qt_balanced = query_triangle(main_edge, main_edge, main_edge)
    qt_balanced_time = galley(qt_balanced, faq_optimizer=greedy, ST=ST, dbconn=dbconn, verbose=verbosity)
    qt_balanced_time = galley(qt_balanced, faq_optimizer=greedy, ST=ST, dbconn=dbconn, verbose=verbosity)
    println("Balanced Triangle [$ST]: ", qt_balanced_time)
    time_dict["balanced triangle"][string(ST) * "_duckdb"] = qt_balanced_time

    qp_balanced = query_path(main_edge, main_edge, main_edge, main_edge)
    qp_balanced_time = galley(qp_balanced, faq_optimizer=greedy, ST=ST, dbconn=dbconn, verbose=verbosity)
    qp_balanced_time = galley(qp_balanced, faq_optimizer=greedy, ST=ST, dbconn=dbconn, verbose=verbosity)
    println("Balanced Path [$ST]: ", qp_balanced_time)
    time_dict["balanced path"][string(ST) * "_duckdb"] = qp_balanced_time

    qb_balanced = query_bowtie(main_edge, main_edge, main_edge, main_edge, main_edge, main_edge)
    qb_balanced_time = galley(qb_balanced, faq_optimizer=greedy, ST=ST, dbconn=dbconn, verbose=verbosity)
    qb_balanced_time = galley(qb_balanced, faq_optimizer=greedy, ST=ST, dbconn=dbconn, verbose=verbosity)
    println("Balanced Bowtie [$ST]: ", qb_balanced_time)
    time_dict["balanced bowtie"][string(ST) * "_duckdb"] = qb_balanced_time
end

dbconn = DBInterface.connect(DuckDB.DB, ":memory:")
vertices, edges = load_dataset("Experiments/Data/Subgraph_Data/yeast/yeast.graph", NaiveStats, dbconn,subgraph_matching_data=true)
main_edge = edges[0]

qt_balanced = query_triangle(main_edge, main_edge, main_edge)
qt_balanced_time = galley(qt_balanced, faq_optimizer=naive, ST=NaiveStats, dbconn=dbconn, verbose=verbosity)
qt_balanced_time = galley(qt_balanced, faq_optimizer=naive, ST=NaiveStats, dbconn=dbconn, verbose=verbosity)
println("Balanced Triangle [DuckDB]: ", qt_balanced_time)
time_dict["balanced triangle"]["DuckDB"] = qt_balanced_time

qp_balanced = query_path(main_edge, main_edge, main_edge, main_edge)
qp_balanced_time = galley(qp_balanced, faq_optimizer=naive, ST=NaiveStats, dbconn=dbconn, verbose=verbosity)
qp_balanced_time = galley(qp_balanced, faq_optimizer=naive, ST=NaiveStats, dbconn=dbconn, verbose=verbosity)
println("Balanced Path [DuckDB]: ", qp_balanced_time)
time_dict["balanced path"]["DuckDB"] = qp_balanced_time

qb_balanced = query_bowtie(main_edge, main_edge, main_edge, main_edge, main_edge, main_edge)
qb_balanced_time = galley(qb_balanced, faq_optimizer=naive, ST=NaiveStats, dbconn=dbconn, verbose=verbosity)
qb_balanced_time = galley(qb_balanced, faq_optimizer=naive, ST=NaiveStats, dbconn=dbconn, verbose=verbosity)
println("Balanced Bowtie [DuckDB]: ", qb_balanced_time)
time_dict["balanced bowtie"]["DuckDB"] = qb_balanced_time

for qt in keys(time_dict)
    println("Query Type: $(qt)")
    for ST in keys(time_dict[qt])
        println("   $(ST): $(time_dict[qt][ST])")
    end
end

Xs = String[]
execute_times = Float64[]
opt_times = Float64[]
groups = String[]
for qt in keys(time_dict)
    for group in keys(time_dict[qt])
        push!(Xs, qt)
        push!(execute_times, time_dict[qt][group].execute_time)
        push!(opt_times, time_dict[qt][group].opt_time)
        push!(groups, group)
    end
end


using StatsPlots
ENV["GKSwstype"]="100"
gbplot = StatsPlots.groupedbar(Xs,
                                execute_times,
                                group = groups,
                                yscale =:log10,
                                ylims=[10^-4, 10^2],
                                legend = :outertopleft,
                                size = (1400, 600))
xlabel!(gbplot, "Query Type")
ylabel!(gbplot, "Execution Time")
savefig(gbplot, "Experiments/Figures/yeast_small_execute.png")

gbplot = StatsPlots.groupedbar(Xs,
                                opt_times,
                                group = groups,
                                yscale =:log10,
                                ylims=[10^-4, 1],
                                legend = :outertopleft,
                                size = (1400, 600))
xlabel!(gbplot, "Query Type")
ylabel!(gbplot, "Optimization Time")
savefig(gbplot, "Experiments/Figures/yeast_small_opt.png")
