
using Test
using Galley
using Galley: canonicalize, insert_statistics!, get_reduce_query, AnnotatedQuery, reduce_idx, cost_of_reduce, greedy_aq_to_plan
using Finch

A = Tensor(Dense(Sparse(Element(0.0))), fsprand(5, 5, .2))

@testset "Annotated Queries" begin
    @testset "get_reduce_query" begin
        chain_expr = Query(:out, Materialize(Aggregate(+, :i, :j, :k, MapJoin(*, Input(A, :i, :j), Input(A, :j, :k)))))
        aq = AnnotatedQuery(chain_expr, NaiveStats)
        println(aq.idx_op)
        query, new_aq = reduce_idx(Index(:i), aq)
        expected_expr = Aggregate(+, :i, Input(A, :i, :j))
        @test query.expr == expected_expr

        query, new_aq = reduce_idx(Index(:j), aq)
        expected_expr = Aggregate(+, :i, :j, :k, MapJoin(*, Input(A, :i, :j), Input(A, :j, :k)))
        @test query.expr == expected_expr

        query, new_aq = reduce_idx(Index(:k), aq)
        expected_expr = Aggregate(+, :k, Input(A, :j, :k))
        @test query.expr == expected_expr

        # Check that we don't push aggregates past operations which don't distribute over them.
        chain_expr = Query(:out, Materialize(Aggregate(+, :i, :j, :k, MapJoin(max, Input(A, :i, :j), Input(A, :j, :k)))))
        aq = AnnotatedQuery(chain_expr, NaiveStats)

        query, new_aq = reduce_idx(Index(:i), aq)
        expected_expr = Aggregate(+, :i, :j, :k, MapJoin(max, Input(A, :i, :j), Input(A, :j, :k)))
        @test query.expr == expected_expr


        # Check that we respect aggregates' position in the exression
        chain_expr = Query(:out, Materialize(Aggregate(+, :j, :k, MapJoin(max, Aggregate(+, :i, Input(A, :i, :j)), Input(A, :j, :k)))))
        aq = AnnotatedQuery(chain_expr, NaiveStats)

        query, new_aq = reduce_idx(Index(:i), aq)
        expected_expr = Aggregate(+, :i, Input(A, :i, :j))
        @test query.expr == expected_expr
    end


    @testset "get_reduce_query" begin


    end
end


chain_expr = Query(:out, Materialize(Aggregate(max, :i, :j, :k, MapJoin(+, Input(A, :i, :j), Input(A, :j, :k)))))
println(galley(chain_expr; verbose=4))
