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
    factors = Set(Factor[Factor(e1, Set(IndexExpr[i, j]), Set(IndexExpr[i, j]), false, deepcopy(e1.stats), 1),
                     Factor(e2, Set(IndexExpr[j, k]), Set(IndexExpr[j, k]), false, deepcopy(e2.stats), 2),
                     Factor(e3, Set(IndexExpr[k, i]), Set(IndexExpr[k, i]), false, deepcopy(e3.stats), 3),
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
            output[] += e1[gallop(i), j] * e2[gallop(k), j] * e3[gallop(k), i]
        end
    end
end


function finch_triangle_follow(e1, e2, e3)
    e1 = e1.args[2]
    e2 = e2.args[2]
    e3 = e3.args[2]
    e4 = Tensor(DenseLevel(SparseHashLevel{1}(Element(0.0))))
    @finch (e4 .= 0; for j=_, i=_ e4[i,j] = e3[i,j]; end)
    output = Finch.Scalar(0.0)
    return @elapsed @finch begin
        output .= 0
        for j=_, i=_, k=_
            output[] += e1[i, j] * e2[k,  j] * e4[follow(k), follow(i)]
        end
    end
end



function query_mm(e1, e2)
    i = IndexExpr("i")
    j = IndexExpr("j")
    k = IndexExpr("k")
    e1 = e1[i,j]
    e2 = e2[j,k]
    factors = Set(Factor[Factor(e1, Set(IndexExpr[i, j]), Set(IndexExpr[i, j]), false, deepcopy(e1.stats), 1),
                     Factor(e2, Set(IndexExpr[j, k]), Set(IndexExpr[j, k]), false, deepcopy(e2.stats), 2),
    ])
    faq = FAQInstance(*, +, Set{IndexExpr}(), Set{IndexExpr}([i, j, k]), factors)
    return faq
end

function finch_mm(e1, e2)
    e1 = e1.args[2]
    e2 = e2.args[2]
    E1 = Finch.Tensor(Dense(SparseList(Element(0.0), size(e1)[1]), size(e1)[2]))
    E2 = Finch.Tensor(Dense(SparseList(Element(0.0), size(e2)[1]), size(e2)[2]))
    Finch.copyto!(E1, e1)
    Finch.copyto!(E2, e2)

    output = Finch.Scalar(0.0)
    return @elapsed @finch begin
        output .= 0
        for j=_, i=_, k=_
            output[] += E1[i,j] * E2[k,j]
        end
    end
end


function finch_mm2(e1, e2)
    e1 = e1.args[2]
    e2 = e2.args[2]
    E1 = Finch.Tensor(Dense(SparseList(Element(0.0), size(e1)[1]), size(e1)[2]))
    E2 = Finch.Tensor(Dense(SparseList(Element(0.0), size(e2)[1]), size(e2)[2]))
    Finch.copyto!(E1, e1)
    Finch.copyto!(E2, e2)

    I = Tensor(Dense(Element(0.0)))
    output = Finch.Scalar(0.0)
    return @elapsed begin
        @finch begin
            I .= 0
            for j=_, i=_
                I[j] += E1[i,j]
            end
        end
        @finch begin
            output .= 0
            for j=_, k=_
                output[] += I[j] * E2[k,j]
            end
        end
    end
end


function finch_mm_proper_dcsc(e1, e2)
    e1 = e1.args[2]
    e2 = e2.args[2]
    E1 = Finch.Tensor(SparseList(SparseList(Element(0.0), size(e1)[1]), size(e1)[2]))
    E2 = Finch.Tensor(SparseList(SparseList(Element(0.0), size(e2)[1]), size(e2)[2]))
    Finch.copyto!(E1, e1)
    Finch.copyto!(E2, e2)

    output = Finch.Tensor(SparseHash{1}(SparseHash{1}(Element(0.0))))
    return @elapsed @finch begin
        output .= 0
        for k=_, j=_, i=_
            output[i, j] += E1[i, k] * E2[j, k]
        end
    end
end


function finch_mm_proper(e1, e2)
    e1 = e1.args[2]
    e2 = e2.args[2]
    E1 = Finch.Tensor(Dense(SparseList(Element(0.0), size(e1)[1]), size(e1)[2]))
    E2 = Finch.Tensor(Dense(SparseList(Element(0.0), size(e2)[1]), size(e2)[2]))
    Finch.copyto!(E1, e1)
    Finch.copyto!(E2, e2)

    output = Finch.Tensor(Dense(SparseHash{1}(Element(0.0))))
    return @elapsed @finch begin
        output .= 0
        for k=_, j=_, i=_
            output[i, j] += E1[i, k] * E2[j, k]
        end
    end
end

function finch_mm_proper_gustavsons(e1, e2)
    e1 = e1.args[2]
    e2 = swizzle(e2.args[2], 2, 1)
    E1 = Finch.Tensor(Dense(SparseList(Element(0.0), size(e1)[1]), size(e1)[2]))
    E2 = Finch.Tensor(Dense(SparseList(Element(0.0), size(e2)[1]), size(e2)[2]))
    Finch.copyto!(E1, e1)
    Finch.copyto!(E2, e2)

    C = Tensor(Dense(SparseList(Element(0.0))))
    w = Tensor(SparseByteMap(Element(0.0)))
    return @elapsed @finch begin
        C .= 0
        for j=_
            w .= 0
            for k=_, i=_; w[i] += E1[i, follow(k)] * E2[gallop(k), j] end
            for i=_; C[i, j] = w[i] end
        end
    end
end

function finch_mm_proper_row_major(e1, e2)
    e1 = e1.args[2]
    e2 = swizzle(e2.args[2], 2, 1)
    E1 = Finch.Tensor(Dense(SparseList(Element(0.0), size(e1)[1]), size(e1)[2]))
    E2 = Finch.Tensor(Dense(SparseList(Element(0.0), size(e2)[1]), size(e2)[2]))
    Finch.copyto!(E1, e1)
    Finch.copyto!(E2, e2)

    C = Tensor(Dense(SparseHash{1}(Element(0.0))))
    w = Tensor(SparseByteMap(Element(0.0)))
    return @elapsed @finch begin
        C .= 0
        for j=_, k=_, i=_;
            C[i, j] += E1[i, k] * E2[k, j]
        end
    end
end

function finch_mm_proper_inner(e1, e2)
    e1 = e1.args[2]
    e2 = e2.args[2]

    z = Finch.default(e1) * Finch.default(e2) + false
    C = Tensor(Dense(SparseList(Element(z))))
    w = Tensor(SparseDict(SparseDict(Element(z))))
    AT = Tensor(Dense(SparseList(Element(z))))
    @finch mode=fastfinch (w .= 0; for k=_, i=_; w[k, i] = e1[i, k] end)
    @finch mode=fastfinch (AT .= 0; for i=_, k=_; AT[k, i] = w[k, i] end)
    @finch (C .= 0; for j=_, i=_, k=_; C[i, j] += AT[k, gallop(i)] * e2[k, gallop(j)] end)
    return C
end

function query_mm_proper(e1, e2)
    i = IndexExpr("i")
    j = IndexExpr("j")
    k = IndexExpr("k")
    e1 = e1[i, k]
    e2 = e2[j, k]
    factors = Set(Factor[Factor(e1, Set(IndexExpr[k, i]), Set(IndexExpr[k, i]), false, deepcopy(e1.stats), 1),
                     Factor(e2, Set(IndexExpr[k, j]), Set(IndexExpr[k, j]), false, deepcopy(e2.stats), 2),
    ])
    faq = FAQInstance(*, +, Set{IndexExpr}([i,j]), Set{IndexExpr}([i, j, k]), factors, [i, j])
    return faq
end

verbosity=3
vertices, edges = load_dataset("Experiments/Data/Subgraph_Data/aids/aids.txt", NaiveStats, nothing)
main_edge = edges[0]

t_finch_follow = finch_triangle_follow(main_edge, main_edge, main_edge)
t_finch_follow = finch_triangle_follow(main_edge, main_edge, main_edge)
qt_balanced = query_triangle(main_edge, main_edge, main_edge)
t_duckdb = duckdb_compute_faq(qt_balanced).time
t_finch = finch_triangle(main_edge, main_edge, main_edge)
t_finch = finch_triangle(main_edge, main_edge, main_edge)
t_finch_gallop = finch_triangle_gallop(main_edge, main_edge, main_edge)
t_finch_gallop = finch_triangle_gallop(main_edge, main_edge, main_edge)
println("t_duckdb: $(t_duckdb)")
println("t_finch: $(t_finch)")
println("t_finch_gallop: $(t_finch_gallop)")
println("t_finch_follow: $(t_finch_follow)")

mm_balanced = query_mm(main_edge, main_edge)
mm_duckdb = duckdb_compute_faq(mm_balanced).time
mm_finch = finch_mm(main_edge, main_edge)
mm_finch = finch_mm(main_edge, main_edge)
mm_finch_materialize = finch_mm2(main_edge, main_edge)
mm_finch_materialize = finch_mm2(main_edge, main_edge)
println("mmsum_duckdb: $(mm_duckdb)")
println("mmsum_finch: $(mm_finch)")
println("mmsum_finch_materialize: $(mm_finch_materialize)")

mm_balanced = query_mm_proper(main_edge, main_edge)
mm_duckdb = duckdb_compute_faq(mm_balanced).time
mm_finch = finch_mm_proper(main_edge, main_edge)
mm_finch = finch_mm_proper(main_edge, main_edge)
#mm_finch_inner = finch_mm_proper_inner(main_edge, main_edge)
#mm_finch_inner = finch_mm_proper_inner(main_edge, main_edge)
#mm_finch_dcsc = finch_mm_proper_dcsc(main_edge, main_edge)
#mm_finch_dcsc = finch_mm_proper_dcsc(main_edge, main_edge)
mm_finch_gustavsons = finch_mm_proper_gustavsons(main_edge, main_edge)
mm_finch_gustavsons = finch_mm_proper_gustavsons(main_edge, main_edge)
println("mm_duckdb: $(mm_duckdb)")
println("mm_finch: $(mm_finch)")
#println("mm_finch_inner: $(mm_finch_inner)")
#println("mm_finch_dcsc: $(mm_finch_dcsc)")
println("mm_finch_gustavsons: $(mm_finch_gustavsons)")
