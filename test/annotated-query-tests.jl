
using Test
using Galley
using Galley: canonicalize, insert_statistics!, get_reduce_query, AnnotatedQuery, reduce_idx!, cost_of_reduce, greedy_query_to_plan
using Finch

@testset verbose = true "Annotated Queries" begin
    A = Tensor(Dense(Sparse(Element(0.0))), fsprand(5, 5, .2))

    @testset "get_reduce_query" begin
        chain_expr = Query(:out, Materialize(Aggregate(+, :i, :j, :k, MapJoin(*, Input(A, :i, :j, "a1"), Input(A, :j, :k, "a2")))))
        aq = AnnotatedQuery(chain_expr, NaiveStats)
        query = reduce_idx!(Index(:i), aq)
        expected_expr = Aggregate(+, :i, Input(A, :i, :j, "a1"))
        @test query.expr == expected_expr

        aq = AnnotatedQuery(chain_expr, NaiveStats)
        query = reduce_idx!(Index(:j), aq)
        expected_expr = Aggregate(+, :i, :j, :k, MapJoin(*, Input(A, :i, :j, "a1"), Input(A, :j, :k, "a2")))
        @test query.expr == expected_expr

        aq = AnnotatedQuery(chain_expr, NaiveStats)
        query = reduce_idx!(Index(:k), aq)
        expected_expr = Aggregate(+, :k, Input(A, :j, :k, "a2"))
        @test query.expr == expected_expr

        # Check that we don't push aggregates past operations which don't distribute over them.
        chain_expr = Query(:out, Materialize(Aggregate(+, :i, :j, :k, MapJoin(max, Input(A, :i, :j, "a1"), Input(A, :j, :k, "a2")))))
        aq = AnnotatedQuery(chain_expr, NaiveStats)
        query = reduce_idx!(Index(:i), aq)
        expected_expr = Aggregate(+, :i, :j, :k, MapJoin(max, Input(A, :i, :j, "a1"), Input(A, :j, :k, "a2")))
        @test query.expr == expected_expr

        # Check that we respect aggregates' position in the exression
        chain_expr = Query(:out, Materialize(Aggregate(+, :j, :k, MapJoin(max, Aggregate(+, :i, Input(A, :i, :j, "a1")), Input(A, :j, :k, "a2")))))
        aq = AnnotatedQuery(chain_expr, NaiveStats)
        query = reduce_idx!(Index(:i), aq)
        expected_expr = Aggregate(+, :i, Input(A, :i, :j, "a1"))
        @test query.expr == expected_expr

    end
end

nothing
