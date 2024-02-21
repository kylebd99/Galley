using Galley
using Finch
using BenchmarkTools
using Galley: initmax, _calc_dc_from_structure, IndexExpr

include("../Experiments.jl")

function query_triangle(e1, e2, e3)
    i = IndexExpr("i")
    j = IndexExpr("j")
    k = IndexExpr("k")
    e1 = e1[i,j]
    e2 = e2[j,k]
    e3 = e3[k,i]
    factors = Set(Factor[Factor(e1, Set(IndexExpr[i, j]), Set(IndexExpr[i, j]), false, deepcopy(e1.stats)),
                     Factor(e2, Set(IndexExpr[j, k]), Set(IndexExpr[j, k]), false, deepcopy(e2.stats)),
                     Factor(e3, Set(IndexExpr[k, i]), Set(IndexExpr[k, i]), false, deepcopy(e3.stats)),
    ])
    faq = FAQInstance(*, +, Set{IndexExpr}(), Set{IndexExpr}([i, j, k]), factors)
    return faq
end


function query_path(e1, e2, e3, e4)
    i = IndexExpr("i")
    j = IndexExpr("j")
    k = IndexExpr("k")
    l = IndexExpr("l")
    m = IndexExpr("m")
    e1 = e1[i,j]
    e2 = e2[j,k]
    e3 = e3[k,l]
    e4 = e4[l,m]
    factors = Set(Factor[Factor(e1, Set(IndexExpr[i, j]), Set(IndexExpr[i, j]), false, deepcopy(e1.stats)),
                     Factor(e2, Set(IndexExpr[j, k]), Set(IndexExpr[j, k]), false, deepcopy(e2.stats)),
                     Factor(e3, Set(IndexExpr[k, l]), Set(IndexExpr[k, l]), false, deepcopy(e3.stats)),
                     Factor(e4, Set(IndexExpr[l, m]), Set(IndexExpr[l, m]), false, deepcopy(e4.stats)),
    ])

    faq = FAQInstance(*, +, Set{IndexExpr}(), Set{IndexExpr}([i, j, k, l, m]), factors)
    return faq
end



function query_bowtie(e1, e2, e3, e4, e5, e6)
    i = IndexExpr("i")
    j = IndexExpr("j")
    k = IndexExpr("k")
    l = IndexExpr("l")
    m = IndexExpr("m")
    e1 = e1[i,j]
    e2 = e2[j,k]
    e3 = e3[i,k]
    e4 = e4[l,m]
    e5 = e5[l,k]
    e6 = e6[m,k]
    factors = Set(Factor[Factor(e1, Set(IndexExpr[i, j]), Set(IndexExpr[i, j]), false, deepcopy(e1.stats)),
                     Factor(e2, Set(IndexExpr[j, k]), Set(IndexExpr[j, k]), false, deepcopy(e2.stats)),
                     Factor(e3, Set(IndexExpr[i, k]), Set(IndexExpr[i, k]), false, deepcopy(e3.stats)),
                     Factor(e4, Set(IndexExpr[l, m]), Set(IndexExpr[l, m]), false, deepcopy(e4.stats)),
                     Factor(e5, Set(IndexExpr[l, k]), Set(IndexExpr[l, k]), false, deepcopy(e5.stats)),
                     Factor(e6, Set(IndexExpr[m, k]), Set(IndexExpr[m, k]), false, deepcopy(e6.stats)),
    ])

    faq = FAQInstance(*, +, Set{IndexExpr}(), Set{IndexExpr}([i, j, k, l, m]), factors)
    return faq
end



time_dict = Dict("balanced triangle"=>Dict(),
                "unbalanced triangle"=>Dict(),
                "balanced path"=>Dict(),
                "unbalanced path"=>Dict(),
                "balanced bowtie"=>Dict(),
                "unbalanced bowtie"=>Dict(), )

verbosity=3
vertices, edges = load_dataset("Experiments/Data/Subgraph_Data/aids/aids.txt", NaiveStats)
main_edge = edges[0]

qt_balanced = query_triangle(main_edge, main_edge, main_edge)
duckdb_compute_faq(qt_balanced)

#=
galley(qt_balanced, faq_optimizer=greedy, verbose=0)
qt_balanced_time = @elapsed galley(qt_balanced, faq_optimizer=greedy, verbose=verbosity)
println("Balanced Triangle: ", qt_balanced_time)

qt_unbalanced = query_triangle(edges[0], edges[1], edges[2])
galley(qt_unbalanced, faq_optimizer=greedy, verbose=0)
qt_unbalanced_time = @elapsed galley(qt_unbalanced, faq_optimizer=greedy, verbose=verbosity)
println("Unbalanced Triangle: ", qt_unbalanced_time)

qp_balanced = query_path(main_edge, main_edge, main_edge, main_edge)
galley(qp_balanced, faq_optimizer=greedy, verbose=0)
qp_balanced_time = @elapsed galley(qp_balanced, faq_optimizer=greedy, verbose=verbosity)
println("Balanced Path: ", qp_balanced_time)

qp_unbalanced = query_path(edges[0], edges[1], edges[2], edges[3])
galley(qp_unbalanced, faq_optimizer=greedy, verbose=0)
qp_unbalanced_time = @elapsed galley(qp_unbalanced, faq_optimizer=greedy, verbose=verbosity)
println("Unbalanced Path: ", qp_unbalanced_time)

qb_balanced = query_bowtie(main_edge, main_edge, main_edge, main_edge, main_edge, main_edge)
galley(qb_balanced, faq_optimizer=greedy, verbose=verbosity)
qb_balanced_time = @elapsed galley(qb_balanced, faq_optimizer=greedy, verbose=verbosity)
println("Balanced Bowtie: ", qb_balanced_time)

qb_unbalanced = query_bowtie(edges[0], edges[0], edges[0], edges[3], edges[3], edges[3])
galley(qb_unbalanced, faq_optimizer=greedy, verbose=0)
qb_unbalanced_time = @elapsed galley(qb_unbalanced, faq_optimizer=greedy, verbose=verbosity)
println("Unbalanced Bowtie: ", qb_unbalanced_time)
 =#
