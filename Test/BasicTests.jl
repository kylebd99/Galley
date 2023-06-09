using Test
using Finch

include("../Source/Spartan.jl")
verbose = 0


@testset "matrix operations" begin
    @testset "2x2 matrices, element-wise mult" begin
        a_matrix = [1 0; 0 1]
        a_fiber = Fiber(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(a_fiber, a_matrix)
        a = InputTensor(a_fiber)
        b_matrix = [0 1; 1 0]
        b_fiber = Fiber(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(b_fiber, b_matrix)
        b = InputTensor(b_fiber)
        c = OutTensor()
        c["i","j"] = a["i","j"] * b["i","j"]
        spartan_matrix = spartan(c, optimize=false, verbose=verbose)
        correct_matrix = a_matrix .* b_matrix
        @test spartan_matrix == correct_matrix
    end

    @testset "2x2 matrices, element-wise add" begin
        a_matrix = [1 0; 0 1]
        a_fiber = Fiber(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(a_fiber, a_matrix)
        a = InputTensor(a_fiber)
        b_matrix = [0 1; 1 0]
        b_fiber = Fiber(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(b_fiber, b_matrix)
        b = InputTensor(b_fiber)
        c = OutTensor()
        c["i","j"] = a["i","j"] + b["i","j"]
        spartan_matrix = spartan(c, optimize=false, verbose=verbose)
        correct_matrix = a_matrix .+ b_matrix
        @test spartan_matrix == correct_matrix
    end

    @testset "2x2 matrices, element-wise custom" begin
        f(x,y) = min(x,y)
        a_matrix = [1 0; 0 1]
        a_fiber = Fiber(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(a_fiber, a_matrix)
        a = InputTensor(a_fiber)
        b_matrix = [0 1; 1 0]
        b_fiber = Fiber(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(b_fiber, b_matrix)
        b = InputTensor(b_fiber)
        c = OutTensor()
        c["i","j"] = f(a["i","j"], b["i","j"])
        spartan_matrix = spartan(c, optimize=false, verbose=verbose)
        correct_matrix = [0 0; 0 0]
        @test spartan_matrix == correct_matrix
    end

    @testset "2x2 matrices, element-wise mult, reverse dim" begin
        a_matrix = [1 1; 0 0]
        a_fiber = Fiber(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(a_fiber, a_matrix)
        a = InputTensor(a_fiber)
        b_matrix = [1 1; 0 0]
        b_fiber = Fiber(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(b_fiber, b_matrix)
        b = InputTensor(b_fiber)
        c = OutTensor()
        c["i","j"] = a["i","j"] * b["j","i"]
        spartan_matrix = spartan(c, optimize=false, verbose=verbose)
        correct_matrix = a_matrix .* (b_matrix')
        @test spartan_matrix == correct_matrix
    end

    @testset "2x2 matrices, element-wise mult, reverse output" begin
        a_matrix = [1 1; 0 0]
        a_fiber = Fiber(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(a_fiber, a_matrix)
        a = InputTensor(a_fiber)
        b_matrix = [1 1; 0 0]
        b_fiber = Fiber(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(b_fiber, b_matrix)
        b = InputTensor(b_fiber)
        c = OutTensor()
        c["j","i"] = a["i","j"] * b["i","j"]
        spartan_matrix = spartan(c, optimize=false, verbose=verbose)
        correct_matrix = [1 0; 1 0]
        @test spartan_matrix == correct_matrix
    end

    @testset "2x2 matrices, matrix mult" begin
        a_matrix = [1 1; 0 0]
        a_fiber = Fiber(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(a_fiber, a_matrix)
        a = InputTensor(a_fiber)
        b_matrix = [1 1; 0 0]
        b_fiber = Fiber(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(b_fiber, b_matrix)
        b = InputTensor(b_fiber)
        c = OutTensor()
        c["i", "j"] = ReduceDim(+, ["j"], a["i","j"] * b["j", "k"])
        spartan_matrix = spartan(c, optimize=false, verbose=verbose)
        correct_matrix = a_matrix * b_matrix
        @test spartan_matrix == correct_matrix
    end


    @testset "2x2 matrices, matrix mult, custom add" begin
        f(args...) = +(0, args...)
        a_matrix = [1 1; 0 0]
        a_fiber = Fiber(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(a_fiber, a_matrix)
        a = InputTensor(a_fiber)
        b_matrix = [1 1; 0 0]
        b_fiber = Fiber(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(b_fiber, b_matrix)
        b = InputTensor(b_fiber)
        c = OutTensor()
        c["i", "j"] = ReduceDim(f, ["j"], a["i","j"] * b["j", "k"])
        spartan_matrix = spartan(c, optimize=false, verbose=verbose)
        correct_matrix = a_matrix * b_matrix
        @test spartan_matrix == correct_matrix
    end
end
