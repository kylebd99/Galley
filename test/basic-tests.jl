

verbose = 0

@testset "matrix operations" begin
    @testset "2x2 matrices, element-wise mult" begin
        a_matrix = [1 0; 0 1]
        a_fiber = Tensor(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(a_fiber, a_matrix)
        a = InputTensor(a_fiber)
        b_matrix = [0 1; 1 0]
        b_fiber = Tensor(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(b_fiber, b_matrix)
        b = InputTensor(b_fiber)
        c = OutTensor()
        c["i","j"] = a["i","j"] * b["i","j"]
        galley_matrix = galley(c, optimize=true, verbose=verbose)
        correct_matrix = a_matrix .* b_matrix
        @test galley_matrix == correct_matrix
    end

    @testset "2x2 matrices, element-wise add" begin
        a_matrix = [1 0; 0 1]
        a_fiber = Tensor(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(a_fiber, a_matrix)
        a = InputTensor(a_fiber)
        b_matrix = [0 1; 1 0]
        b_fiber = Tensor(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(b_fiber, b_matrix)
        b = InputTensor(b_fiber)
        c = OutTensor()
        c["i","j"] = a["i","j"] + b["i","j"]
        galley_matrix = galley(c, optimize=true, verbose=verbose)
        correct_matrix = a_matrix .+ b_matrix
        @test galley_matrix == correct_matrix
    end

    @testset "2x2 matrices, element-wise custom" begin
        f(x,y) = min(x,y)
        a_matrix = [1 0; 0 1]
        a_fiber = Tensor(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(a_fiber, a_matrix)
        a = InputTensor(a_fiber)
        b_matrix = [0 1; 1 0]
        b_fiber = Tensor(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(b_fiber, b_matrix)
        b = InputTensor(b_fiber)
        c = OutTensor()
        c["i","j"] = f(a["i","j"], b["i","j"])
        galley_matrix = galley(c, optimize=true, verbose=verbose)
        correct_matrix = [0 0; 0 0]
        @test galley_matrix == correct_matrix
    end

    @testset "2x2 matrices, element-wise mult, reverse input" begin
        a_matrix = [1 1; 0 0]
        a_fiber = Tensor(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(a_fiber, a_matrix)
        a = InputTensor(a_fiber)
        b_matrix = [1 1; 0 0]
        b_fiber = Tensor(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(b_fiber, b_matrix)
        b = InputTensor(b_fiber)
        c = OutTensor()
        c["i","j"] = a["i","j"] * b["j","i"]
        galley_matrix = galley(c, optimize=true, verbose=verbose)
        correct_matrix = a_matrix .* (b_matrix')
        @test galley_matrix == correct_matrix
    end

    @testset "100x100 matrices, element-wise mult, reverse output" begin
        a_matrix = sprand(Bool, 100, 100, .01)
        a_fiber = Tensor(SparseList(SparseList(Element(0), 100), 100))
        copyto!(a_fiber, a_matrix)
        a = InputTensor(a_fiber)
        b_matrix = sprand(Bool, 100, 100, .01)
        b_fiber = Tensor(SparseList(SparseList(Element(0), 100), 100))
        copyto!(b_fiber, b_matrix)
        b = InputTensor(b_fiber)
        c = OutTensor()
        c["j","i"] = a["i","j"] * b["i","j"]
        galley_matrix = galley(c, optimize=true, verbose=verbose)
        correct_matrix = (a_matrix.*b_matrix)'
        @test galley_matrix == correct_matrix
    end

    @testset "100x100 matrices, matrix mult" begin
        a_matrix = sprand(Bool, 100, 100, .01)
        a_fiber = Tensor(SparseList(SparseList(Element(0), 100), 100))
        copyto!(a_fiber, a_matrix)
        a = InputTensor(a_fiber)
        b_matrix = sprand(Bool, 100, 100, .01)
        b_fiber = Tensor(SparseList(SparseList(Element(0), 100), 100))
        copyto!(b_fiber, b_matrix)
        b = InputTensor(b_fiber)
        c = OutTensor()
        c["i", "k"] = ∑(["j"], a["i","j"] * b["j", "k"])
        galley_matrix = galley(c, optimize=true, verbose=verbose)
        correct_matrix = a_matrix * b_matrix
        @test galley_matrix == correct_matrix
    end


    @testset "100x100 matrices, matrix mult, custom add" begin
        f(args...) = +(0, args...)
        a_matrix = sprand(Bool, 100, 100, .1)
        a_fiber = Tensor(SparseList(SparseList(Element(0), 100), 100))
        copyto!(a_fiber, a_matrix)
        a = InputTensor(a_fiber)
        b_matrix = sprand(Bool, 100, 100, .1)
        b_fiber = Tensor(SparseList(SparseList(Element(0), 100), 100))
        copyto!(b_fiber, b_matrix)
        b = InputTensor(b_fiber)
        c = OutTensor()
        c["i", "k"] = Agg(f, ["j"], a["i","j"] * b["j", "k"])
        galley_matrix = galley(c, optimize=true, verbose=verbose)
        correct_matrix = a_matrix * b_matrix
        @test galley_matrix == correct_matrix
    end


    @testset "100x100 matrices, full sum" begin
        a_matrix = sprand(Bool, 100, 100, .01)
        a_fiber = Tensor(SparseList(SparseList(Element(0), 100), 100))
        copyto!(a_fiber, a_matrix)
        a = InputTensor(a_fiber)
        c = OutTensor()
        c[] = ∑(["i", "j"], a["i","j"])
        galley_matrix = galley(c, optimize=true, verbose=verbose)
        correct_matrix = sum(a_matrix)
        @test galley_matrix() == correct_matrix
    end

    @testset "100x100 matrices, multi-line, matrix mult" begin
        a_matrix = sprand(Bool, 100, 100, .1)
        a_fiber = Tensor(SparseList(SparseList(Element(0), 100), 100))
        copyto!(a_fiber, a_matrix)
        a = InputTensor(a_fiber)
        b_matrix = sprand(Bool, 100, 100, .1)
        b_fiber = Tensor(SparseList(SparseList(Element(0), 100), 100))
        copyto!(b_fiber, b_matrix)
        b = InputTensor(b_fiber)
        c_matrix = sprand(Bool, 100, 100, .1)
        c_fiber = Tensor(SparseList(SparseList(Element(0), 100), 100))
        copyto!(c_fiber, c_matrix)
        c = InputTensor(c_fiber)
        d = OutTensor()
        e = OutTensor()
        d["i", "k"] = ∑("j", a["i","j"] * b["j", "k"])
        e["i", "l"] = ∑("k", d["i","k"] * c["k", "l"])
        galley_matrix = galley(e, optimize=true, verbose=verbose)
        correct_matrix = a_matrix * b_matrix * c_matrix
        @test galley_matrix == correct_matrix
    end

    @testset "100x100 matrices, multi-line, reuse, matrix mult" begin
        a_matrix = sprand(Bool, 100, 100, .1)
        a_fiber = Tensor(SparseList(SparseList(Element(0), 100), 100))
        copyto!(a_fiber, a_matrix)
        a = InputTensor(a_fiber)
        b_matrix = sprand(Bool, 100, 100, .1)
        b_fiber = Tensor(SparseList(SparseList(Element(0), 100), 100))
        copyto!(b_fiber, b_matrix)
        b = InputTensor(b_fiber)
        d = OutTensor()
        e = OutTensor()
        d["i", "k"] = ∑("j", a["i","j"] * b["j", "k"])
        e["i", "l"] = ∑("k", d["i","k"] * d["k", "l"])
        verbose= 0
        galley_matrix = galley(e, optimize=true, verbose=verbose)
        d_matrix = a_matrix * b_matrix
        correct_matrix = d_matrix * d_matrix
        @test galley_matrix == correct_matrix
    end


    @testset "100x100 matrices, diagonal mult" begin
        a_matrix = sprand(Bool, 100, 100, .1)
        a_fiber = Tensor(SparseList(SparseList(Element(0), 100), 100))
        copyto!(a_fiber, a_matrix)
        a = InputTensor(a_fiber)
        b_matrix = sprand(Bool, 100, 100, .1)
        b_fiber = Tensor(SparseList(SparseList(Element(0), 100), 100))
        copyto!(b_fiber, b_matrix)
        b = InputTensor(b_fiber)
        d = OutTensor()
        d["i"] =  a["i","i"] * b["i", "i"]
        galley_matrix = galley(d, optimize = true, verbose=verbose)
        correct_matrix = spzeros(100)
        for i in 1:100
            correct_matrix[i] = a_matrix[i,i] * b_matrix[i,i]
        end
        @test galley_matrix == correct_matrix
    end
end

nothing
