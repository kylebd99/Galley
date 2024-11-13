

@testset verbose = true "Annotated Queries" begin
    A = Tensor(Dense(Sparse(Element(0.0))), fsprand(5, 5, .2))

    @testset "get_reduce_query" begin
        chain_expr = Query(:out, Materialize(Aggregate(+, 0, :i, :j, :k, MapJoin(*, Input(A, :i, :j, "a1"), Input(A, :j, :k, "a2")))))
        aq = AnnotatedQuery(chain_expr, NaiveStats)
        query = reduce_idx!(:i, aq)
        expected_expr = Aggregate(+, 0, :i, Input(A, :i, :j, "a1"))
        @test query.expr == expected_expr

        aq = AnnotatedQuery(chain_expr, NaiveStats)
        query = reduce_idx!(:j, aq)
        expected_expr = Aggregate(+, 0, :i, :j, :k, MapJoin(*, Input(A, :i, :j, "a1"), Input(A, :j, :k, "a2")))
        @test query.expr == expected_expr

        aq = AnnotatedQuery(chain_expr, NaiveStats)
        query = reduce_idx!(:k, aq)
        expected_expr = Aggregate(+, 0, :k, Input(A, :j, :k, "a2"))
        @test query.expr == expected_expr

        # Check that we don't push aggregates past operations which don't distribute over them.
        chain_expr = Query(:out, Materialize(Aggregate(+, 0, :i, :j, :k, MapJoin(max, Input(A, :i, :j, "a1"), Input(A, :j, :k, "a2")))))
        aq = AnnotatedQuery(chain_expr, NaiveStats)
        query = reduce_idx!(:i, aq)
        expected_expr = Aggregate(+, 0, :i, :j, :k, MapJoin(max, Input(A, :i, :j, "a1"), Input(A, :j, :k, "a2")))
        @test query.expr == expected_expr

        # Check that we respect aggregates' position in the exression
        chain_expr = Query(:out, Materialize(Aggregate(+, 0, :j, :k, MapJoin(max, Aggregate(+, 0, :i, Input(A, :i, :j, "a1")), Input(A, :j, :k, "a2")))))
        aq = AnnotatedQuery(chain_expr, NaiveStats)
        query = reduce_idx!(:i, aq)
        expected_expr = Aggregate(+, 0, :i, Input(A, :i, :j, "a1"))
        @test query.expr == expected_expr
    end
end

nothing
