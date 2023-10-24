# This file tests the hyper tree decomposition functionality.


@testset "FAQInstance to HTD" begin

end

# The conversion of an HTD to a logical plan should be a fairly straightforward process.
@testset "HTD to LogicalPlan" begin

    @testset "matrix multiplication" begin
        i = IndexExpr("i")
        j = IndexExpr("j")
        k = IndexExpr("k")

        a_matrix = [1 0; 0 1]
        a_fiber = Fiber(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(a_fiber, a_matrix)
        a_tensor = InputTensor(a_fiber)[i, j]
        a_factor = Factor(a_tensor, Set([i, j]), Set([i, j]), false, TensorStats([i, j], a_fiber, nothing))
        b_matrix = [0 1; 1 0]
        b_fiber = Fiber(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(b_fiber, b_matrix)
        b_tensor = InputTensor(b_fiber)[j, k]
        b_factor = Factor(b_tensor, Set([j, k]), Set([j, k]), false, TensorStats([j,k], b_fiber, nothing))

        bag = Bag([a_factor, b_factor], Set([i, j, k]), Set([i, k]), Vector{Bag}())
        htd = HyperTreeDecomposition(*, +, Set([i, k]), bag)
        correct_plan = Aggregate(+, IndexExpr[j], MapJoin(*, a_tensor, b_tensor))
        @test decomposition_to_logical_plan(htd) == correct_plan
    end
end

# The conversion of an HTD to a logical plan should be a fairly straightforward process.
@testset "HTD to Output" begin

    @testset "matrix multiplication" begin
        i = IndexExpr("i")
        j = IndexExpr("j")
        k = IndexExpr("k")

        a_matrix = [1 0; 0 1]
        a_fiber = Fiber(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(a_fiber, a_matrix)
        a_tensor = InputTensor(a_fiber)[i, j]
        a_factor = Factor(a_tensor, Set([i, j]), Set([i, j]), false, TensorStats([i, j], a_fiber, nothing))
        b_matrix = [0 1; 1 0]
        b_fiber = Fiber(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(b_fiber, b_matrix)
        b_tensor = InputTensor(b_fiber)[j, k]
        b_factor = Factor(b_tensor, Set([j, k]), Set([j, k]), false, TensorStats([j,k], b_fiber, nothing))

        bag = Bag([a_factor, b_factor], Set([i, j, k]), Set([i, k]), Vector{Bag}())
        htd = HyperTreeDecomposition(*, +, Set([i, k]), bag)
        correct_plan = Aggregate(+, IndexExpr[j], MapJoin(*, a_tensor, b_tensor))
        plan = decomposition_to_logical_plan(htd)
        global_index_order = [i, j, k]
        plan = insert_global_orders(plan, global_index_order)
        plan = fill_in_stats(plan, global_index_order)
        tensor_kernel = expr_to_kernel(plan, global_index_order, verbose = verbose)
        galley_matrix = execute_tensor_kernel(tensor_kernel, verbose = verbose)
        correct_matrix = a_matrix * b_matrix
        @test galley_matrix == correct_matrix
    end
end

@testset "FAQ to Output" begin

    @testset "matrix multiplication" begin
        i = IndexExpr("i")
        j = IndexExpr("j")
        k = IndexExpr("k")

        a_matrix = [1 0; 0 1]
        a_fiber = Fiber(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(a_fiber, a_matrix)
        a_tensor = InputTensor(a_fiber)[i, j]
        a_factor = Factor(a_tensor, Set([i, j]), Set([i, j]), false, TensorStats([i, j], a_fiber, nothing))
        b_matrix = [0 1; 1 0]
        b_fiber = Fiber(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(b_fiber, b_matrix)
        b_tensor = InputTensor(b_fiber)[j, k]
        b_factor = Factor(b_tensor, Set([j, k]), Set([j, k]), false, TensorStats([j,k], b_fiber, nothing))

        faq = FAQInstance(*, +, Set([i,k]), Set([i,j,k]), [a_factor, b_factor])
        galley_matrix = galley(faq, verbose = verbose)
        correct_matrix = a_matrix * b_matrix
        @test galley_matrix == correct_matrix
    end
end
