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

        bag = Bag(nothing, [a_factor, b_factor], Set([i, j, k]), Set([i, k]), Vector{Bag}())
        htd = HyperTreeDecomposition(*, +, Set([i, k]), bag)
        correct_plan = Aggregate(+, IndexExpr[j], MapJoin(*, a_tensor, b_tensor))
        @test decomposition_to_logical_plan(htd) == correct_plan
    end
end
