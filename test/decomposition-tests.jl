# This file tests the hyper tree decomposition functionality.


@testset "FAQInstance to HTD" begin

end

# The conversion of an HTD to a logical plan should be a fairly straightforward process.
@testset verbose = true  "HTD to LogicalPlan" begin
    @testset "matrix multiplication" begin
        i = IndexExpr("i")
        j = IndexExpr("j")
        k = IndexExpr("k")

        a_matrix = [1 0; 0 1]
        a_data = Tensor(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(a_data, a_matrix)
        a_tensor = InputTensor(a_data)[i, j]
        a_factor = Factor(a_tensor, Set([i, j]), Set([i, j]), false, NaiveStats([i, j], a_data), 1)
        b_matrix = [0 1; 1 0]
        b_data = Tensor(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(b_data, b_matrix)
        b_tensor = InputTensor(b_data)[j, k]
        b_factor = Factor(b_tensor, Set([j, k]), Set([j, k]), false, NaiveStats([j,k], b_data), 2)

        bag = Bag(*, +, Set([a_factor, b_factor]), Set([i, j, k]), Set([i, k]), Set{Bag}(), 1)
        htd = HyperTreeDecomposition(*, +, Set([i, k]), bag, nothing)
        correct_plan_1 = Aggregate(+, Set{IndexExpr}([j]), MapJoin(*, a_tensor, b_tensor))
        correct_plan_2 = Aggregate(+, Set{IndexExpr}([j]), MapJoin(*, b_tensor, a_tensor))
        @test decomposition_to_logical_plan(htd) == correct_plan_1 || decomposition_to_logical_plan(htd) == correct_plan_2
    end
end

# The conversion of an HTD to a logical plan should be a fairly straightforward process.
@testset verbose = true  "HTD to Output" begin
    @testset "matrix multiplication" begin
        i = IndexExpr("i")
        j = IndexExpr("j")
        k = IndexExpr("k")

        a_matrix = [1 0; 0 1]
        a_data = Tensor(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(a_data, a_matrix)
        a_tensor = InputTensor(a_data)[i, j]
        a_factor = Factor(a_tensor, Set([i, j]), Set([i, j]), false, NaiveStats([i, j], a_data), 1)
        b_matrix = [0 1; 1 0]
        b_data = Tensor(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(b_data, b_matrix)
        b_tensor = InputTensor(b_data)[j, k]
        b_factor = Factor(b_tensor, Set([j, k]), Set([j, k]), false, NaiveStats([j,k], b_data), 2)
        bag = Bag(*, +, Set([a_factor, b_factor]), Set([i, j, k]), Set([i, k]), Set{Bag}(), 1)
        htd = HyperTreeDecomposition(*, +, Set([i, k]), bag, nothing)
        correct_plan = Aggregate(+, Set{IndexExpr}([j]), MapJoin(*, a_tensor, b_tensor))
        plan = decomposition_to_logical_plan(htd)
        _recursive_insert_stats!(plan)
        output_order = [i, k]
        tensor_kernel = expr_to_kernel(plan, output_order)
        correct_matrix = a_matrix * b_matrix
        @test execute_tensor_kernel(tensor_kernel) == correct_matrix
    end
end

@testset verbose = true "FAQ to Output" begin
    @testset "matrix multiplication" begin
        i = IndexExpr("i")
        j = IndexExpr("j")
        k = IndexExpr("k")

        m,n = 10, 10
        p = .05
        T = Int64
        a_matrix =  abs.(sprand(T, m, n, p) .% 100)
        a_data = Tensor(SparseList(SparseList(Element(zero(T)), m), n))
        copyto!(a_data, a_matrix)
        a_tensor = InputTensor(a_data)[i, j]
        a_factor = Factor(a_tensor, Set([i, j]), Set([i, j]), false, NaiveStats([i, j], a_data), 1)
        b_matrix =  abs.(sprand(T, m, n, p) .% 100)
        b_data = Tensor(SparseList(SparseList(Element(zero(T)), m), n))
        copyto!(b_data, b_matrix)
        b_tensor = InputTensor(b_data)[j, k]
        b_factor = Factor(b_tensor, Set([j, k]), Set([j, k]), false, NaiveStats([j,k], b_data), 2)
        faq = FAQInstance(*, +, Set([i,k]), Set([i,j,k]), Set([a_factor, b_factor]), [i, k])
        correct_matrix = a_matrix * b_matrix
        @test galley(faq; faq_optimizer = naive).value == correct_matrix
        @test galley(faq; faq_optimizer = greedy).value == correct_matrix
        @test galley(faq; faq_optimizer = hypertree_width).value == correct_matrix
    end

    @testset "matrix multiplication chain" begin
        i = IndexExpr("i")
        j = IndexExpr("j")
        k = IndexExpr("k")
        l = IndexExpr("l")

        m,n = 10, 10
        p = .05
        T = Int64
        a_matrix = abs.(sprand(T, m, n, p) .% 100)
        a_data = Tensor(SparseList(SparseList(Element(zero(T)), m), n))
        copyto!(a_data, a_matrix)
        a_tensor = InputTensor(a_data)[i, j]
        a_factor = Factor(a_tensor, Set([i, j]), Set([i, j]), false, NaiveStats([i, j], a_data), 1)
        b_matrix = abs.(sprand(T, m, n, p) .% 100)
        b_data = Tensor(SparseList(SparseList(Element(zero(T)), m), n))
        copyto!(b_data, b_matrix)
        b_tensor = InputTensor(b_data)[j, k]
        b_factor = Factor(b_tensor, Set([j, k]), Set([j, k]), false, NaiveStats([j,k], b_data), 2)
        c_matrix =  abs.(sprand(T, m, n, p) .% 100)
        c_data = Tensor(SparseList(SparseList(Element(zero(T)), m), n))
        copyto!(c_data, c_matrix)
        c_tensor = InputTensor(c_data)[k, l]
        c_factor = Factor(c_tensor, Set([k, l]), Set([k, l]), false, NaiveStats([k, l], c_data), 3)
        faq = FAQInstance(*, +, Set([i,l]), Set([i,j,k,l]), Set([a_factor, b_factor, c_factor]), [i, l])
        correct_matrix = a_matrix * b_matrix * c_matrix
        @test galley(faq; faq_optimizer = naive).value == correct_matrix
        @test galley(faq; faq_optimizer = greedy).value == correct_matrix
        @test galley(faq; faq_optimizer = hypertree_width).value == correct_matrix
    end

    @testset "element-wise matrix multiplication" begin
        i = IndexExpr("i")
        j = IndexExpr("j")

        m,n = 10, 10
        p = .05
        T = Int64
        a_matrix = abs.(sprand(T, m, n, p) .% 100)
        a_data = Tensor(SparseList(SparseList(Element(zero(T)), m), n))
        copyto!(a_data, a_matrix)
        a_tensor = InputTensor(a_data)[i, j]
        a_factor = Factor(a_tensor, Set([i, j]), Set([i, j]), false, NaiveStats([i, j], a_data), 1)
        b_matrix = abs.(sprand(T, m, n, p) .% 100)
        b_data = Tensor(SparseList(SparseList(Element(zero(T)), m), n))
        copyto!(b_data, b_matrix)
        b_tensor = InputTensor(b_data)[i, j]
        b_factor = Factor(b_tensor, Set([i, j]), Set([i, j]), false, NaiveStats([i, j], b_data), 2)
        c_matrix = abs.(sprand(T, m, n, p) .% 100)
        c_data = Tensor(SparseList(SparseList(Element(zero(T)), m), n))
        copyto!(c_data, c_matrix)
        c_tensor = InputTensor(c_data)[i, j]
        c_factor = Factor(c_tensor, Set([i, j]), Set([i, j]), false, NaiveStats([i, j], c_data), 3)
        faq = FAQInstance(*, +, Set([i, j]), Set([i,j]), Set([a_factor, b_factor, c_factor]), [i, j])
        correct_matrix = a_matrix .* b_matrix .* c_matrix
        @test galley(faq; faq_optimizer = naive).value == correct_matrix
        @test galley(faq; faq_optimizer = greedy).value == correct_matrix
        @test galley(faq; faq_optimizer = hypertree_width).value == correct_matrix
    end
end
