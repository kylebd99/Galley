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



function finch_triangle(e1, e2, e3)
    e1 = e1.args[2]
    e2 = e2.args[2]
    e3 = e3.args[2]
    output = Finch.Scalar(0.0)
    return @elapsed @finch begin
        output .= 0
        for j=_, i=_, k=_
            output[] += e1[i,j] * e2[k,j] * e3[k, i]
        end
    end
end

function finch_triangle_gallop(e1, e2, e3)
    e1 = e1.args[2]
    e2 = e2.args[2]
    e3 = e3.args[2]
    output = Finch.Scalar(0.0)
    return @elapsed @finch begin
        output .= 0
        for j=_, i=_, k=_
            output[] += e1[gallop(i), gallop(j)] * e2[gallop(k), gallop(j)] * e3[gallop(k), gallop(i)]
        end
    end
end

function query_mm(e1, e2)
    i = IndexExpr("i")
    j = IndexExpr("j")
    k = IndexExpr("k")
    e1 = e1[i,j]
    e2 = e2[j,k]
    factors = Set(Factor[Factor(e1, Set(IndexExpr[i, j]), Set(IndexExpr[i, j]), false, deepcopy(e1.stats)),
                     Factor(e2, Set(IndexExpr[j, k]), Set(IndexExpr[j, k]), false, deepcopy(e2.stats)),
    ])
    faq = FAQInstance(*, +, Set{IndexExpr}(), Set{IndexExpr}([i, j, k]), factors)
    return faq
end

function finch_mm(e1, e2)
    e1 = e1.args[2]
    e2 = e2.args[2]
    output = Finch.Scalar(0.0)
    return @elapsed @finch begin
        output .= 0
        for j=_, i=_, k=_
            output[] += e1[i,j] * e2[k,j]
        end
    end
end


function finch_mm2(e1, e2)
    e1 = e1.args[2]
    e2 = e2.args[2]
    I = Tensor(Dense(Element(0.0), size(e1)[1]))
    output = Finch.Scalar(0.0)
    return @elapsed begin
        @finch begin
            I .= 0
            for j=_, i=_
                I[j] += e1[i,j]
            end
        end
        @finch begin
            output .= 0
            for j=_, k=_
                output[] += I[j] * e2[k,j]
            end
        end
    end
end


verbosity=3
vertices, edges = load_dataset("Experiments/Data/Subgraph_Data/aids/aids.txt", NaiveStats)
main_edge = edges[0]

qt_balanced = query_triangle(main_edge, main_edge, main_edge)
t_duckdb = duckdb_compute_faq(qt_balanced).time
t_finch = finch_triangle(main_edge, main_edge, main_edge)
t_finch = finch_triangle(main_edge, main_edge, main_edge)
t_finch_gallop = finch_triangle_gallop(main_edge, main_edge, main_edge)
t_finch_gallop = finch_triangle_gallop(main_edge, main_edge, main_edge)
println("t_duckdb: $(t_duckdb)")
println("t_finch: $(t_finch)")
println("t_finch_gallop: $(t_finch_gallop)")

mm_balanced = query_mm(main_edge, main_edge)
mm_duckdb = duckdb_compute_faq(mm_balanced).time
mm_finch = finch_mm(main_edge, main_edge)
mm_finch = finch_mm(main_edge, main_edge)
mm_finch_materialize = finch_mm2(main_edge, main_edge)
mm_finch_materialize = finch_mm2(main_edge, main_edge)
println("mm_duckdb: $(mm_duckdb)")
println("mm_finch: $(mm_finch)")
println("mm_finch_materialize: $(mm_finch_materialize)")



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
