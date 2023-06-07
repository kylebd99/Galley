using Test
using Finch

include("../Source/Spartan.jl")
verbose = 0


@testset "matrix operations" begin
    @testset "2x2 matrices, element-wise mult" begin
        a_matrix = [1 0; 0 1]
        a_fiber = Fiber(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(a_fiber, a_matrix)
        a = InputTensor("a", ["i", "j"], a_fiber)
        b_matrix = [0 1; 1 0]
        b_fiber = Fiber(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(b_fiber, b_matrix)
        b = InputTensor("b", ["i", "j"], b_fiber)
        spartan_matrix = spartan(:(MapJoin($*, $a, $b)), optimize=false, verbose=verbose)
        correct_matrix = a_matrix .* b_matrix
        @test spartan_matrix == correct_matrix
    end

    @testset "2x2 matrices, element-wise add" begin
        a_matrix = [1 0; 0 1]
        a_fiber = Fiber(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(a_fiber, a_matrix)
        a = InputTensor("a", ["i", "j"], a_fiber)
        b_matrix = [0 1; 1 0]
        b_fiber = Fiber(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(b_fiber, b_matrix)
        b = InputTensor("b", ["i", "j"], b_fiber)
        spartan_matrix = spartan(:(MapJoin($+, $a, $b)), optimize=false, verbose=verbose)
        correct_matrix = a_matrix .+ b_matrix
        @test spartan_matrix == correct_matrix
    end

    @testset "2x2 matrices, element-wise custom" begin
        f(x,y) = min(x,y)
        a_matrix = [1 0; 0 1]
        a_fiber = Fiber(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(a_fiber, a_matrix)
        a = InputTensor("a", ["i", "j"], a_fiber)
        b_matrix = [0 1; 1 0]
        b_fiber = Fiber(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(b_fiber, b_matrix)
        b = InputTensor("b", ["i", "j"], b_fiber)
        spartan_matrix = spartan(:(MapJoin($f, $a, $b)), optimize=false, verbose=verbose)
        correct_matrix = [0 0; 0 0]
        @test spartan_matrix == correct_matrix
    end

    @testset "2x2 matrices, element-wise mult, reverse dim" begin
        a_matrix = [1 1; 0 0]
        a_fiber = Fiber(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(a_fiber, a_matrix)
        a = InputTensor("a", ["i", "j"], a_fiber)
        b_matrix = [1 1; 0 0]
        b_fiber = Fiber(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(b_fiber, b_matrix)
        b = InputTensor("b", ["j", "i"], b_fiber)
        spartan_matrix = spartan(:(MapJoin($*, $a, $b)), optimize=false, verbose=verbose)
        correct_matrix = a_matrix .* (b_matrix')
        @test spartan_matrix == correct_matrix
    end

    @testset "2x2 matrices, element-wise mult, reverse output" begin
        a_matrix = [1 1; 0 0]
        a_fiber = Fiber(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(a_fiber, a_matrix)
        a = InputTensor("a", ["i", "j"], a_fiber)
        b_matrix = [1 1; 0 0]
        b_fiber = Fiber(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(b_fiber, b_matrix)
        b = InputTensor("b", ["i", "j"], b_fiber)
        spartan_matrix = spartan(:(Reorder(MapJoin($*, $a, $b), $["j", "i"])), optimize=false, verbose=verbose)
        correct_matrix = [1 0; 1 0]
        @test spartan_matrix == correct_matrix
    end

    @testset "2x2 matrices, matrix mult" begin
        a_matrix = [1 1; 0 0]
        a_fiber = Fiber(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(a_fiber, a_matrix)
        a = InputTensor("a", ["i", "j"], a_fiber)
        b_matrix = [1 1; 0 0]
        b_fiber = Fiber(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(b_fiber, b_matrix)
        b = InputTensor("b", ["j", "k"], b_fiber)
        spartan_matrix = spartan(:(ReduceDim($+, $(["j"]), MapJoin($*, $a, $b))), optimize=false, verbose=2)
        correct_matrix = a_matrix * b_matrix
        @test spartan_matrix == correct_matrix
    end


    @testset "2x2 matrices, matrix mult, custom add" begin
        f(args...) = +(0, args...)
        a_matrix = [1 1; 0 0]
        a_fiber = Fiber(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(a_fiber, a_matrix)
        a = InputTensor("a", ["i", "j"], a_fiber)
        b_matrix = [1 1; 0 0]
        b_fiber = Fiber(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(b_fiber, b_matrix)
        b = InputTensor("b", ["j", "k"], b_fiber)
        spartan_matrix = spartan(:(ReduceDim($f, $(["j"]), MapJoin($*, $a, $b))), optimize=false, verbose=2)
        correct_matrix = a_matrix * b_matrix
        @test spartan_matrix == correct_matrix
    end

end