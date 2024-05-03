
using Galley: estimate_nnz, reduce_tensor_stats, condense_stats!

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
        condense_stats!(stat)
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
        condense_stats!(stat)
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
        condense_stats!(stat)
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
        condense_stats!(stat)
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
        condense_stats!(stat)
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
        condense_stats!(stat)
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
        condense_stats!(stat)
        reduce_stats = reduce_tensor_stats(+, Set([i]), stat)
        @test estimate_nnz(reduce_stats) == 5
    end
end

nothing
