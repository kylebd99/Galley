
using Galley: estimate_nnz, reduce_tensor_stats, condense_stats!, merge_tensor_stats

@testset "NaiveStats" begin


end


@testset verbose = true "DCStats" begin


    @testset "Single Tensor Card" begin
        i = IndexExpr("i")
        j = IndexExpr("j")
        dims = Dict(i=>1000, j=>1000)
        def = TensorDef(Set([i,j]), dims, 0.0, nothing, nothing, nothing)
        dcs = Set([DC(Set([i]), Set([j]), 5),
                 DC(Set([j]), Set([i]), 25),
                 DC(Set(), Set([i, j]), 50),
                ])
        stat = DCStats(def, dcs)
        @test estimate_nnz(stat) == 50
    end

    @testset "1 Join DC Card" begin
        i = IndexExpr("i")
        j = IndexExpr("j")
        k = IndexExpr("k")
        dims = Dict(i=>1000, j=>1000, k=>1000)
        def = TensorDef(Set([i,j,k]), dims, 0.0, nothing, nothing, nothing)
        dcs = Set([
                 DC(Set([j]), Set([k]), 5),
                 DC(Set(), Set([i, j]), 50),
                ])
        stat = DCStats(def, dcs)
        @test estimate_nnz(stat) == 50*5
    end

    @testset "2 Join DC Card" begin
        i = IndexExpr("i")
        j = IndexExpr("j")
        k = IndexExpr("k")
        l = IndexExpr("l")
        dims = Dict(i=>1000, j=>1000, k=>1000, l=>1000)
        def = TensorDef(Set([i,j,k,l]), dims, 0.0, nothing, nothing, nothing)
        dcs = Set([
                DC(Set(), Set([i, j]), 50),
                DC(Set([j]), Set([k]), 5),
                DC(Set([k]), Set([l]), 5),
                ])
        stat = DCStats(def, dcs)
        @test estimate_nnz(stat) == 50*5*5
    end

    @testset "Triangle DC Card" begin
        i = IndexExpr("i")
        j = IndexExpr("j")
        k = IndexExpr("k")
        dims = Dict(i=>1000, j=>1000, k=>1000)
        def = TensorDef(Set([i,j,k]), dims, 0.0, nothing, nothing, nothing)
        dcs = Set([
                DC(Set(), Set([i, j]), 50),
                DC(Set([i]), Set([j]), 5),
                DC(Set([j]), Set([i]), 5),
                DC(Set(), Set([j, k]), 50),
                DC(Set([j]), Set([k]), 5),
                DC(Set([k]), Set([j]), 5),
                DC(Set(), Set([i, k]), 50),
                DC(Set([i]), Set([k]), 5),
                DC(Set([k]), Set([i]), 5),
                ])
        stat = DCStats(def, dcs)
        @test estimate_nnz(stat) == 50*5
    end

    @testset "Triangle-Small DC Card" begin
        i = IndexExpr("i")
        j = IndexExpr("j")
        k = IndexExpr("k")
        dims = Dict(i=>1000, j=>1000, k=>1000)
        def = TensorDef(Set([i,j,k]), dims, 0.0, nothing, nothing, nothing)
        # In this version, |R(i,j)| = 1
        dcs = Set([
                DC(Set(), Set([i, j]), 1),
                DC(Set([i]), Set([j]), 1),
                DC(Set([j]), Set([i]), 1),
                DC(Set(), Set([j, k]), 50),
                DC(Set([j]), Set([k]), 5),
                DC(Set([k]), Set([j]), 5),
                DC(Set(), Set([i, k]), 50),
                DC(Set([i]), Set([k]), 5),
                DC(Set([k]), Set([i]), 5),
                ])
        stat = DCStats(def, dcs)
        @test estimate_nnz(stat) == 1*5
    end

    @testset "Full Reduce DC Card" begin
        i = IndexExpr("i")
        j = IndexExpr("j")
        k = IndexExpr("k")
        dims = Dict(i=>1000, j=>1000, k=>1000)
        def = TensorDef(Set([i,j,k]), dims, 0.0, nothing, nothing, nothing)
        dcs = Set([
                DC(Set(), Set([i, j]), 50),
                DC(Set([i]), Set([j]), 5),
                DC(Set([j]), Set([i]), 5),
                DC(Set(), Set([j, k]), 50),
                DC(Set([j]), Set([k]), 5),
                DC(Set([k]), Set([j]), 5),
                DC(Set(), Set([i, k]), 50),
                DC(Set([i]), Set([k]), 5),
                DC(Set([k]), Set([i]), 5),
                ])
        stat = DCStats(def, dcs)
        reduce_stats = reduce_tensor_stats(+, Set([i,j,k]), stat)
        @test estimate_nnz(reduce_stats) == 1
    end

    @testset "1-Attr Reduce DC Card" begin
        i = IndexExpr("i")
        j = IndexExpr("j")
        k = IndexExpr("k")
        dims = Dict(i=>1000, j=>1000, k=>1000)
        def = TensorDef(Set([i,j,k]), dims, 0.0, nothing, nothing, nothing)
        dcs = Set([
                    DC(Set(), Set([i, j]), 1),
                    DC(Set([i]), Set([j]), 1),
                    DC(Set([j]), Set([i]), 1),
                    DC(Set(), Set([j, k]), 50),
                    DC(Set([j]), Set([k]), 5),
                    DC(Set([k]), Set([j]), 5),
                    DC(Set(), Set([i, k]), 50),
                    DC(Set([i]), Set([k]), 5),
                    DC(Set([k]), Set([i]), 5),
                ])
        stat = DCStats(def, dcs)
        condense_stats!(stat)
        reduce_stats = reduce_tensor_stats(+, Set([i, j]), stat)
        @test estimate_nnz(reduce_stats) == 5
    end

    @testset "2-Attr Reduce DC Card" begin
        i = IndexExpr("i")
        j = IndexExpr("j")
        k = IndexExpr("k")
        dims = Dict(i=>1000, j=>1000, k=>1000)
        def = TensorDef(Set([i,j,k]), dims, 0.0, nothing, nothing, nothing)
        dcs = Set([
                    DC(Set(), Set([i, j]), 1),
                    DC(Set([i]), Set([j]), 1),
                    DC(Set([j]), Set([i]), 1),
                    DC(Set(), Set([j, k]), 50),
                    DC(Set([j]), Set([k]), 5),
                    DC(Set([k]), Set([j]), 5),
                    DC(Set(), Set([i, k]), 50),
                    DC(Set([i]), Set([k]), 5),
                    DC(Set([k]), Set([i]), 5),
                ])
        stat = DCStats(def, dcs)
        reduce_stats = reduce_tensor_stats(+, Set([i]), stat)
        @test estimate_nnz(reduce_stats) == 5
    end

    @testset "1D Disjunction DC Card" begin
        dims = Dict(:i=>1000)
        def = TensorDef(Set([:i]), dims, 0.0, nothing, nothing, nothing)
        dcs1 = Set([DC(Set(), Set([:i]), 1),])
        stat1 = DCStats(def, dcs1)
        dcs2 = Set([DC(Set(), Set([:i]), 1),])
        stat2 = DCStats(def, dcs2)
        reduce_stats = merge_tensor_stats(+, stat1, stat2)
        @test estimate_nnz(reduce_stats) == 2
    end

    @testset "2D Disjunction DC Card" begin
        dims = Dict(:i=>1000, :j => 100)
        def = TensorDef(Set([:i, :j]), dims, 0.0, nothing, nothing, nothing)
        dcs1 = Set([DC(Set(), Set([:i, :j]), 1),])
        stat1 = DCStats(def, dcs1)
        dcs2 = Set([DC(Set(), Set([:i, :j]), 1),])
        stat2 = DCStats(def, dcs2)
        merge_stats = merge_tensor_stats(+, stat1, stat2)
        @test estimate_nnz(merge_stats) == 2
    end

    @testset "2D Disjoint Disjunction DC Card" begin
        dims1 = Dict(:i=>1000)
        def1 = TensorDef(Set([:i]), dims1, 0.0, nothing, nothing, nothing)
        dcs1 = Set([DC(Set(), Set([:i]), 5),])
        stat1 = DCStats(def1, dcs1)
        dims2 = Dict(:j => 100)
        def2 = TensorDef(Set([:j]), dims2, 0.0, nothing, nothing, nothing)
        dcs2 = Set([DC(Set(), Set([:j]), 10),])
        stat2 = DCStats(def2, dcs2)
        merge_stats = merge_tensor_stats(+, stat1, stat2)
        @test estimate_nnz(merge_stats) == (10*1000 + 5*100)
    end

    @testset "3D Disjoint Disjunction DC Card" begin
        dims1 = Dict(:i=>1000, :j=>100)
        def1 = TensorDef(Set([:i, :j]), dims1, 0.0, nothing, nothing, nothing)
        dcs1 = Set([DC(Set(), Set([:i, :j]), 5),])
        stat1 = DCStats(def1, dcs1)
        dims2 = Dict(:j => 100, :k=>1000)
        def2 = TensorDef(Set([:j, :k]), dims2, 0.0, nothing, nothing, nothing)
        dcs2 = Set([DC(Set(), Set([:j, :k]), 10),])
        stat2 = DCStats(def2, dcs2)
        merge_stats = merge_tensor_stats(+, stat1, stat2)
        @test estimate_nnz(merge_stats) == (10*1000 + 5*1000)
    end

    @testset "Mixture Disjunction Conjunction DC Card" begin
        dims1 = Dict(:i=>1000, :j=>100)
        def1 = TensorDef(Set([:i, :j]), dims1, 1, nothing, nothing, nothing)
        dcs1 = Set([DC(Set(), Set([:i, :j]), 5),])
        stat1 = DCStats(def1, dcs1)
        dims2 = Dict(:j => 100, :k=>1000)
        def2 = TensorDef(Set([:j, :k]), dims2, 1, nothing, nothing, nothing)
        dcs2 = Set([DC(Set(), Set([:j, :k]), 10),])
        stat2 = DCStats(def2, dcs2)
        dims3 = Dict(:i=>1000, :j => 100, :k=>1000)
        def3 = TensorDef(Set([:i, :j, :k]), dims3, 0.0, nothing, nothing, nothing)
        dcs3 = Set([DC(Set(), Set([:i, :j, :k]), 10),])
        stat3 = DCStats(def3, dcs3)
        merge_stats = merge_tensor_stats(*, stat1, stat2, stat3)
        @test estimate_nnz(merge_stats) == 10
    end

end

nothing
