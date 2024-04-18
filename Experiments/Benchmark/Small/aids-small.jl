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

verbosity = 2

for ST in [DCStats, NaiveStats]
    vertices, edges = load_dataset("Experiments/Data/Subgraph_Data/aids/aids.txt", ST, nothing)
    main_edge = edges[0]

    qt_balanced = query_triangle(main_edge, main_edge, main_edge)
    qt_balanced_time = galley(qt_balanced, ST=ST, faq_optimizer=greedy, verbose=verbosity)
    qt_balanced_time = galley(qt_balanced, ST=ST, faq_optimizer=greedy, verbose=verbosity)
    println("Balanced Triangle [$ST]: ", qt_balanced_time)
    time_dict["balanced triangle"][string(ST)] = qt_balanced_time

    qt_unbalanced = query_triangle(edges[0], edges[1], edges[2])
    qt_unbalanced_time = galley(qt_unbalanced, ST=ST, faq_optimizer=greedy, verbose=verbosity)
    qt_unbalanced_time = galley(qt_unbalanced, ST=ST, faq_optimizer=greedy, verbose=verbosity)
    println("Unbalanced Triangle [$ST]: ", qt_unbalanced_time)
    time_dict["unbalanced triangle"][string(ST)] = qt_unbalanced_time

    qp_balanced = query_path(main_edge, main_edge, main_edge, main_edge)
    qp_balanced_time = galley(qp_balanced, ST=ST, faq_optimizer=greedy, verbose=verbosity)
    qp_balanced_time = galley(qp_balanced, ST=ST, faq_optimizer=greedy, verbose=verbosity)
    println("Balanced Path [$ST]: ", qp_balanced_time)
    time_dict["balanced path"][string(ST)] = qp_balanced_time

    qp_unbalanced = query_path(edges[0], edges[1], edges[2], edges[3])
    qp_unbalanced_time = galley(qp_unbalanced, ST=ST, faq_optimizer=greedy, verbose=verbosity)
    qp_unbalanced_time = galley(qp_unbalanced, ST=ST, faq_optimizer=greedy, verbose=verbosity)
    println("Unbalanced Path [$ST]: ", qp_unbalanced_time)
    time_dict["unbalanced path"][string(ST)] = qp_unbalanced_time

    qb_balanced = query_bowtie(main_edge, main_edge, main_edge, main_edge, main_edge, main_edge)
    qb_balanced_time = galley(qb_balanced, ST=ST, faq_optimizer=greedy, verbose=verbosity)
    qb_balanced_time = galley(qb_balanced, ST=ST, faq_optimizer=greedy, verbose=verbosity)
    println("Balanced Bowtie [$ST]: ", qb_balanced_time)
    time_dict["balanced bowtie"][string(ST)] = qb_balanced_time

    qb_unbalanced = query_bowtie(edges[0], edges[0], edges[0], edges[3], edges[3], edges[3])
    qb_unbalanced_time = galley(qb_unbalanced, ST=ST, faq_optimizer=greedy, verbose=verbosity)
    qb_unbalanced_time = galley(qb_unbalanced, ST=ST, faq_optimizer=greedy, verbose=verbosity)
    println("Unbalanced Bowtie [$ST]: ", qb_unbalanced_time)
    time_dict["unbalanced bowtie"][string(ST)] = qb_unbalanced_time
end

for ST in [DCStats, NaiveStats]
    dbconn = DBInterface.connect(DuckDB.DB, ":memory:")
    vertices, edges = load_dataset("Experiments/Data/Subgraph_Data/aids/aids.txt", ST, dbconn)
    main_edge = edges[0]

    qt_balanced = query_triangle(main_edge, main_edge, main_edge)
    qt_balanced_time = galley(qt_balanced, faq_optimizer=greedy, dbconn=dbconn, ST = ST, verbose=verbosity)
    qt_balanced_time = galley(qt_balanced, faq_optimizer=greedy, dbconn=dbconn, ST = ST, verbose=verbosity)
    println("Balanced Triangle [$ST]: ", qt_balanced_time)
    time_dict["balanced triangle"][string(ST) * "_duckdb"] = qt_balanced_time

    qt_unbalanced = query_triangle(edges[0], edges[1], edges[2])
    qt_unbalanced_time = galley(qt_unbalanced, faq_optimizer=greedy, dbconn=dbconn, ST = ST, verbose=verbosity)
    qt_unbalanced_time = galley(qt_unbalanced, faq_optimizer=greedy, dbconn=dbconn, ST = ST, verbose=verbosity)
    println("Unbalanced Triangle [$ST]: ", qt_unbalanced_time)
    time_dict["unbalanced triangle"][string(ST) * "_duckdb"] = qt_unbalanced_time

    qp_balanced = query_path(main_edge, main_edge, main_edge, main_edge)
    qp_balanced_time = galley(qp_balanced, faq_optimizer=greedy, dbconn=dbconn, ST = ST, verbose=verbosity)
    qp_balanced_time = galley(qp_balanced, faq_optimizer=greedy, dbconn=dbconn, ST = ST, verbose=verbosity)
    println("Balanced Path [$ST]: ", qp_balanced_time)
    time_dict["balanced path"][string(ST) * "_duckdb"] = qp_balanced_time

    qp_unbalanced = query_path(edges[0], edges[1], edges[2], edges[3])
    qp_unbalanced_time = galley(qp_unbalanced, faq_optimizer=greedy, dbconn=dbconn, ST = ST, verbose=verbosity)
    qp_unbalanced_time = galley(qp_unbalanced, faq_optimizer=greedy, dbconn=dbconn, ST = ST, verbose=verbosity)
    println("Unbalanced Path [$ST]: ", qp_unbalanced_time)
    time_dict["unbalanced path"][string(ST) * "_duckdb"] = qp_unbalanced_time

    qb_balanced = query_bowtie(main_edge, main_edge, main_edge, main_edge, main_edge, main_edge)
    qb_balanced_time = galley(qb_balanced, faq_optimizer=greedy, dbconn=dbconn, ST = ST, verbose=verbosity)
    qb_balanced_time = galley(qb_balanced, faq_optimizer=greedy, dbconn=dbconn, ST = ST, verbose=verbosity)
    println("Balanced Bowtie [$ST]: ", qb_balanced_time)
    time_dict["balanced bowtie"][string(ST) * "_duckdb"] = qb_balanced_time

    qb_unbalanced = query_bowtie(edges[0], edges[0], edges[0], edges[3], edges[3], edges[3])
    qb_unbalanced_time = galley(qb_unbalanced, faq_optimizer=greedy, dbconn=dbconn, ST = ST, verbose=verbosity)
    qb_unbalanced_time = galley(qb_unbalanced, faq_optimizer=greedy, dbconn=dbconn, ST = ST, verbose=verbosity)
    println("Unbalanced Bowtie [$ST]: ", qb_unbalanced_time)
    time_dict["unbalanced bowtie"][string(ST) * "_duckdb"] = qb_unbalanced_time
end
verbosity = 0

dbconn = DBInterface.connect(DuckDB.DB, ":memory:")
vertices, edges = load_dataset("Experiments/Data/Subgraph_Data/aids/aids.txt", NaiveStats, dbconn)
main_edge = edges[0]

qt_balanced = query_triangle(main_edge, main_edge, main_edge)
qt_balanced_time = galley(qt_balanced, faq_optimizer=naive, dbconn=dbconn, ST=NaiveStats, verbose=verbosity)
qt_balanced_time = galley(qt_balanced, faq_optimizer=naive, dbconn=dbconn, ST=NaiveStats, verbose=verbosity)
println("Balanced Triangle [DuckDB]: ", qt_balanced_time)
time_dict["balanced triangle"]["DuckDB"] = qt_balanced_time

qt_unbalanced = query_triangle(edges[0], edges[1], edges[2])
qt_unbalanced_time = galley(qt_unbalanced, faq_optimizer=naive, dbconn=dbconn, ST=NaiveStats, verbose=verbosity)
qt_unbalanced_time = galley(qt_unbalanced, faq_optimizer=naive, dbconn=dbconn, ST=NaiveStats, verbose=verbosity)
println("Unbalanced Triangle [DuckDB]: ", qt_unbalanced_time)
time_dict["unbalanced triangle"]["DuckDB"] = qt_unbalanced_time

qp_balanced = query_path(main_edge, main_edge, main_edge, main_edge)
qp_balanced_time = galley(qp_balanced, faq_optimizer=naive, dbconn=dbconn, ST=NaiveStats, verbose=verbosity)
qp_balanced_time = galley(qp_balanced, faq_optimizer=naive, dbconn=dbconn, ST=NaiveStats, verbose=verbosity)
println("Balanced Path [DuckDB]: ", qp_balanced_time)
time_dict["balanced path"]["DuckDB"] = qp_balanced_time

qp_unbalanced = query_path(edges[0], edges[1], edges[2], edges[3])
qp_unbalanced_time = galley(qp_unbalanced, faq_optimizer=naive, dbconn=dbconn, ST=NaiveStats, verbose=verbosity)
qp_unbalanced_time = galley(qp_unbalanced, faq_optimizer=naive, dbconn=dbconn, ST=NaiveStats, verbose=verbosity)
println("Unbalanced Path [DuckDB]: ", qp_unbalanced_time)
time_dict["unbalanced path"]["DuckDB"] = qp_unbalanced_time

qb_balanced = query_bowtie(main_edge, main_edge, main_edge, main_edge, main_edge, main_edge)
qb_balanced_time = galley(qb_balanced, faq_optimizer=naive, dbconn=dbconn, ST=NaiveStats, verbose=verbosity)
qb_balanced_time = galley(qb_balanced, faq_optimizer=naive, dbconn=dbconn, ST=NaiveStats, verbose=verbosity)
println("Balanced Bowtie [DuckDB]: ", qb_balanced_time)
time_dict["balanced bowtie"]["DuckDB"] = qb_balanced_time

qb_unbalanced = query_bowtie(edges[0], edges[0], edges[0], edges[3], edges[3], edges[3])
qb_unbalanced_time = galley(qb_unbalanced, faq_optimizer=naive, dbconn=dbconn, ST=NaiveStats, verbose=verbosity)
qb_unbalanced_time = galley(qb_unbalanced, faq_optimizer=naive, dbconn=dbconn, ST=NaiveStats, verbose=verbosity)
println("Unbalanced Bowtie [DuckDB]: ", qb_unbalanced_time)
time_dict["unbalanced bowtie"]["DuckDB"] = qb_unbalanced_time

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
                                ylims=[10^-3, 1],
                                legend = :outertopleft,
                                size = (1400, 600))
xlabel!(gbplot, "Query Type")
ylabel!(gbplot, "Execution Time")
savefig(gbplot, "Experiments/Figures/aids_small_execute.png")

gbplot = StatsPlots.groupedbar(Xs,
                                opt_times,
                                group = groups,
                                yscale =:log10,
                                ylims=[10^-4, 1],
                                legend = :outertopleft,
                                size = (1400, 600))
xlabel!(gbplot, "Query Type")
ylabel!(gbplot, "Optimization Time")
savefig(gbplot, "Experiments/Figures/aids_small_opt.png")
