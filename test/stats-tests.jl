@testset "NaiveStats" begin


end


@testset "DCStats" begin

    @testset "Single Tensor Card" begin
        i = IndexExpr("i")
        j = IndexExpr("j")
        dims = Dict(i=>1000, j=>1000)
        def = TensorDef(Set([i,j]), dims, 0.0, nothing)
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
        def = TensorDef(Set([i,j,k]), dims, 0.0, nothing)
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
        def = TensorDef(Set([i,j,k,l]), dims, 0.0, nothing)
        dcs = Set([
                DC(Set(), Set([i, j]), 50),
                DC(Set([j]), Set([k]), 5),
                DC(Set([k]), Set([l]), 5),
                ])
        stat = DCStats(def, dcs)
        @test estimate_nnz(stat) == 50*5*5
    end

end
