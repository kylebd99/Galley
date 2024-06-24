using Test
using Galley
using Galley: canonicalize, insert_statistics!, get_reduce_query, AnnotatedQuery, reduce_idx!, cost_of_reduce, greedy_query_to_plan
using Finch

@testset verbose = true "Plan Equality" begin
    A = Tensor(Dense(Sparse(Element(0.0))), fsprand(5, 5, .2))

    @test Input(A, :i, :j, "a1") == Input(A, :i, :j, "a1")
    @test Input(A, :i, :j, "a1") != Input(A, :i, :j, "a2")
    @test Input(A, :i, :j, "a1") != Input(A, :i, :k, "a1")
    @test MapJoin(exp, Input(A, :i, :j, "a1")) == MapJoin(exp, Input(A, :i, :j, "a1"))
    @test MapJoin(exp, Input(A, :i, :j, "a1")) != MapJoin(exp, Input(A, :i, :k, "a1"))
    @test MapJoin(exp, Input(A, :i, :j, "a1")) != MapJoin(+, Input(A, :i, :j, "a1"))
    @test MapJoin(exp, Input(A, :i, :j, "a1")) != MapJoin(exp, Input(A, :i, :j, "a2"))

    @test MapJoin(exp, Input(A, :i, :j, "a1"), Input(A, :i, :j, "a2")) == MapJoin(exp, Input(A, :i, :j, "a1"), Input(A, :i, :j, "a2"))
    @test MapJoin(exp, Input(A, :i, :j, "a1"), Input(A, :i, :j, "a2")) != MapJoin(exp, Input(A, :i, :j, "a2"), Input(A, :i, :j, "a1"))



end

@testset verbose = true "Plan Hash" begin
    A = Tensor(Dense(Sparse(Element(0.0))), fsprand(5, 5, .2))

    @test hash(Input(A, :i, :j, "a1")) == hash(Input(A, :i, :j, "a1"))
    @test hash(Input(A, :i, :j, "a1")) != hash(Input(A, :i, :j, "a2"))
    @test hash(Input(A, :i, :j, "a1")) != hash(Input(A, :i, :k, "a1"))
    @test hash(MapJoin(exp, Input(A, :i, :j, "a1"))) == hash(MapJoin(exp, Input(A, :i, :j, "a1")))
    @test hash(MapJoin(exp, Input(A, :i, :j, "a1"))) != hash(MapJoin(exp, Input(A, :i, :k, "a1")))
    @test hash(MapJoin(exp, Input(A, :i, :j, "a1"))) != hash(MapJoin(+, Input(A, :i, :j, "a1")))
    @test hash(MapJoin(exp, Input(A, :i, :j, "a1"))) != hash(MapJoin(exp, Input(A, :i, :j, "a2")))

    @test hash(MapJoin(exp, Input(A, :i, :j, "a1"), Input(A, :i, :j, "a2"))) == hash(MapJoin(exp, Input(A, :i, :j, "a1"), Input(A, :i, :j, "a2")))
    @test hash(MapJoin(exp, Input(A, :i, :j, "a1"), Input(A, :i, :j, "a2"))) != hash(MapJoin(exp, Input(A, :i, :j, "a2"), Input(A, :i, :j, "a1")))


end
