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
        a_fiber = Fiber(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(a_fiber, a_matrix)
        a_tensor = InputTensor(a_fiber)[i, j]
        a_factor = Factor(a_tensor, Set([i, j]), Set([i, j]), false, TensorStats([i, j], a_fiber))
        b_matrix = [0 1; 1 0]
        b_fiber = Fiber(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(b_fiber, b_matrix)
        b_tensor = InputTensor(b_fiber)[j, k]
        b_factor = Factor(b_tensor, Set([j, k]), Set([j, k]), false, TensorStats([j,k], b_fiber))

        bag = Bag(*, +, Set([a_factor, b_factor]), Set([i, j, k]), Set([i, k]), Set{Bag}())
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
        a_fiber = Fiber!(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(a_fiber, a_matrix)
        a_tensor = InputTensor(a_fiber)[i, j]
        a_factor = Factor(a_tensor, Set([i, j]), Set([i, j]), false, TensorStats([i, j], a_fiber))
        b_matrix = [0 1; 1 0]
        b_fiber = Fiber!(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(b_fiber, b_matrix)
        b_tensor = InputTensor(b_fiber)[j, k]
        b_factor = Factor(b_tensor, Set([j, k]), Set([j, k]), false, TensorStats([j,k], b_fiber))

        bag = Bag(*, +, Set([a_factor, b_factor]), Set([i, j, k]), Set([i, k]), Set{Bag}())
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
        T = Int64
        a_matrix =  sprand(T, m, n, .25)
        a_fiber = Fiber!(SparseList(SparseList(Element(zero(T)), m), n))
        copyto!(a_fiber, a_matrix)
        a_tensor = InputTensor(a_fiber)[i, j]
        a_factor = Factor(a_tensor, Set([i, j]), Set([i, j]), false, TensorStats([i, j], a_fiber))
        b_matrix =  sprand(T, m, n, .25)
        b_fiber = Fiber!(SparseList(SparseList(Element(zero(T)), m), n))
        copyto!(b_fiber, b_matrix)
        b_tensor = InputTensor(b_fiber)[j, k]
        b_factor = Factor(b_tensor, Set([j, k]), Set([j, k]), false, TensorStats([j,k], b_fiber))
        faq = FAQInstance(*, +, Set([i,k]), Set([i,j,k]), Set([a_factor, b_factor]), [i, k])
        correct_matrix = a_matrix * b_matrix
        @test galley(faq; faq_optimizer = naive) == correct_matrix
        @test galley(faq; faq_optimizer = greedy) == correct_matrix
        @test galley(faq; faq_optimizer = hypertree_width) == correct_matrix
    end

    @testset "matrix multiplication chain" begin
        i = IndexExpr("i")
        j = IndexExpr("j")
        k = IndexExpr("k")
        l = IndexExpr("l")

        m,n = 10, 10
        T = Int64
        a_matrix = sprand(T, m, n, .25)
        a_fiber = Fiber!(SparseList(SparseList(Element(zero(T)), m), n))
        copyto!(a_fiber, a_matrix)
        a_tensor = InputTensor(a_fiber)[i, j]
        a_factor = Factor(a_tensor, Set([i, j]), Set([i, j]), false, TensorStats([i, j], a_fiber))
        b_matrix = sprand(T, m, n, .25)
        b_fiber = Fiber!(SparseList(SparseList(Element(zero(T)), m), n))
        copyto!(b_fiber, b_matrix)
        b_tensor = InputTensor(b_fiber)[j, k]
        b_factor = Factor(b_tensor, Set([j, k]), Set([j, k]), false, TensorStats([j,k], b_fiber))
        c_matrix = sprand(T, m, n, .25)
        c_fiber = Fiber!(SparseList(SparseList(Element(zero(T)), m), n))
        copyto!(c_fiber, c_matrix)
        c_tensor = InputTensor(c_fiber)[k, l]
        c_factor = Factor(c_tensor, Set([k, l]), Set([k, l]), false, TensorStats([k, l], c_fiber))
        faq = FAQInstance(*, +, Set([i,l]), Set([i,j,k,l]), Set([a_factor, b_factor, c_factor]), [i, l])
        correct_matrix = a_matrix * b_matrix * c_matrix
        @test galley(faq; faq_optimizer = naive) == correct_matrix
        @test galley(faq; faq_optimizer = greedy) == correct_matrix
        @test galley(faq; faq_optimizer = hypertree_width) == correct_matrix
    end

    @testset "element-wise matrix multiplication" begin
        i = IndexExpr("i")
        j = IndexExpr("j")

        m,n = 10, 10
        T = Int64
        a_matrix = sprand(T, m, n, .25)
        a_fiber = Fiber!(SparseList(SparseList(Element(zero(T)), m), n))
        copyto!(a_fiber, a_matrix)
        a_tensor = InputTensor(a_fiber)[i, j]
        a_factor = Factor(a_tensor, Set([i, j]), Set([i, j]), false, TensorStats([i, j], a_fiber))
        b_matrix = sprand(T, m, n, .25)
        b_fiber = Fiber!(SparseList(SparseList(Element(zero(T)), m), n))
        copyto!(b_fiber, b_matrix)
        b_tensor = InputTensor(b_fiber)[i, j]
        b_factor = Factor(b_tensor, Set([i, j]), Set([i, j]), false, TensorStats([i, j], b_fiber))
        c_matrix = sprand(T, m, n, .25)
        c_fiber = Fiber!(SparseList(SparseList(Element(zero(T)), m), n))
        copyto!(c_fiber, c_matrix)
        c_tensor = InputTensor(c_fiber)[i, j]
        c_factor = Factor(c_tensor, Set([i, j]), Set([i, j]), false, TensorStats([i, j], c_fiber))
        faq = FAQInstance(*, +, Set([i, j]), Set([i,j]), Set([a_factor, b_factor, c_factor]), [i, j])
        correct_matrix = a_matrix .* b_matrix .* c_matrix
        @test galley(faq; faq_optimizer = naive) == correct_matrix
        @test galley(faq; faq_optimizer = greedy) == correct_matrix
        @test galley(faq; faq_optimizer = hypertree_width) == correct_matrix
    end
end
