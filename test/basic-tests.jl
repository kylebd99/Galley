

verbose = 0

@testset "matrix operations" begin
    @testset "2x2 matrices, element-wise mult" begin
        a_matrix = [1 0; 0 1]
        a_data = Tensor(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(a_data, a_matrix)
        a = Input(a_data, :i, :j)
        b_matrix = [0 1; 1 0]
        b_data = Tensor(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(b_data, b_matrix)
        b = Input(b_data, :i, :j)
        q = Query(:out, Materialize(t_sparse_list, t_sparse_list, :i, :j, MapJoin(*, a, b)))
        result = galley(q, verbose=verbose)
        correct_matrix = a_matrix .* b_matrix
        @test result.value == correct_matrix
    end

    @testset "2x2 matrices, element-wise add" begin
        a_matrix = [1 0; 0 1]
        a_data = Tensor(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(a_data, a_matrix)
        a = Input(a_data, :i, :j)
        b_matrix = [0 1; 1 0]
        b_data = Tensor(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(b_data, b_matrix)
        b = Input(b_data, :i, :j)
        q = Query(:out, Materialize(t_sparse_list, t_sparse_list, :i, :j, MapJoin(+, a, b)))
        result = galley(q, verbose=verbose)
        correct_matrix = a_matrix .+ b_matrix
        @test result.value == correct_matrix
    end

    @testset "2x2 matrices, element-wise custom" begin
        f(x,y) = min(x,y)
        a_matrix = [1 0; 0 1]
        a_data = Tensor(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(a_data, a_matrix)
        a = Input(a_data, :i, :j)
        b_matrix = [0 1; 1 0]
        b_data = Tensor(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(b_data, b_matrix)
        b = Input(b_data, :i, :j)
        q = Query(:out, Materialize(t_sparse_list, t_sparse_list, :i, :j, MapJoin(f, a, b)))
        result = galley(q, verbose=verbose)
        correct_matrix = [0 0; 0 0]
        @test result.value == correct_matrix
    end

    @testset "2x2 matrices, element-wise mult, reverse input" begin
        a_matrix = [1 1; 0 0]
        a_data = Tensor(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(a_data, a_matrix)
        a = Input(a_data, :i, :j)
        b_matrix = [1 1; 0 0]
        b_data = Tensor(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(b_data, b_matrix)
        b = Input(b_data, :j, :i)
        q = Query(:out, Materialize(t_sparse_list, t_sparse_list, :i, :j, MapJoin(*, a, b)))
        result = galley(q, verbose=verbose)
        correct_matrix = a_matrix .* (b_matrix')
        @test result.value == correct_matrix
    end

    @testset "100x100 matrices, element-wise mult, reverse output" begin
        a_matrix = sprand(Bool, 100, 100, .01)
        a_data = Tensor(SparseList(SparseList(Element(0), 100), 100))
        copyto!(a_data, a_matrix)
        a = Input(a_data, :i, :j)
        b_matrix = sprand(Bool, 100, 100, .01)
        b_data = Tensor(SparseList(SparseList(Element(0), 100), 100))
        copyto!(b_data, b_matrix)
        b = Input(b_data, :i, :j)
        q = Query(:out, Materialize(t_sparse_list, t_sparse_list, :j, :i, MapJoin(*, a, b)))
        result = galley(q, verbose=verbose)
        correct_matrix = (a_matrix.*b_matrix)'
        @test result.value == correct_matrix
    end

    @testset "100x100 matrices, matrix mult" begin
        a_matrix = sprand(Bool, 100, 100, .01)
        a_data = Tensor(SparseList(SparseList(Element(0), 100), 100))
        copyto!(a_data, a_matrix)
        a = Input(a_data, :i, :j)
        b_matrix = sprand(Bool, 100, 100, .01)
        b_data = Tensor(SparseList(SparseList(Element(0), 100), 100))
        copyto!(b_data, b_matrix)
        b = Input(b_data, :j, :k)
        q = Query(:out, Materialize(t_sparse_list, t_sparse_list, :i, :k, Aggregate(+, :j, MapJoin(*, a, b))))
        result = galley(q, verbose=verbose)
        correct_matrix = a_matrix * b_matrix
        @test result.value == correct_matrix
    end


    @testset "100x100 matrices, matrix mult, custom add" begin
        f(args...) = +(0, args...)
        a_matrix = sprand(Bool, 100, 100, .1)
        a_data = Tensor(SparseList(SparseList(Element(0), 100), 100))
        copyto!(a_data, a_matrix)
        a = Input(a_data, :i, :j)
        b_matrix = sprand(Bool, 100, 100, .1)
        b_data = Tensor(SparseList(SparseList(Element(0), 100), 100))
        copyto!(b_data, b_matrix)
        b = Input(b_data, :j, :k)
        q = Query(:out, Materialize(t_sparse_list, t_sparse_list, :i, :k, Aggregate(f, :j, MapJoin(*, a, b))))
        result = galley(q, verbose=verbose)
        correct_matrix = a_matrix * b_matrix
        @test result.value == correct_matrix
    end


    @testset "100x100 matrices, full sum" begin
        a_matrix = sprand(Bool, 100, 100, .01)
        a_data = Tensor(SparseList(SparseList(Element(0), 100), 100))
        copyto!(a_data, a_matrix)
        a = Input(a_data, :i, :j)
        q = Query(:out, Materialize(Aggregate(+, :i, :j, a)))
        result = galley(q, verbose=verbose)
        correct_matrix = sum(a_matrix)
        @test result.value == correct_matrix
    end
#=
    @testset "100x100 matrices, multi-line, matrix mult" begin
        a_matrix = sprand(Bool, 100, 100, .1)
        a_data = Tensor(SparseList(SparseList(Element(0), 100), 100))
        copyto!(a_data, a_matrix)
        a = InputTensor(a_data)
        b_matrix = sprand(Bool, 100, 100, .1)
        b_data = Tensor(SparseList(SparseList(Element(0), 100), 100))
        copyto!(b_data, b_matrix)
        b = InputTensor(b_data)
        c_matrix = sprand(Bool, 100, 100, .1)
        c_data = Tensor(SparseList(SparseList(Element(0), 100), 100))
        copyto!(c_data, c_matrix)
        c = InputTensor(c_data)
        d = OutTensor()
        e = OutTensor()
        d["i", "k"] = ∑("j", a["i","j"] * b["j", "k"])
        e["i", "l"] = ∑("k", d["i","k"] * c["k", "l"])
        galley_matrix = galley(e, optimize=true, verbose=verbose)
        correct_matrix = a_matrix * b_matrix * c_matrix
        @test galley_matrix == correct_matrix
    end =#

    @testset "100x100 matrices, multi-line, matrix mult" begin
        a_matrix = sprand(Bool, 100, 100, .1)
        a_data = Tensor(SparseList(SparseList(Element(0), 100), 100))
        copyto!(a_data, a_matrix)
        a = Input(a_data, :i, :j)
        b_matrix = sprand(Bool, 100, 100, .1)
        b_data = Tensor(SparseList(SparseList(Element(0), 100), 100))
        copyto!(b_data, b_matrix)
        b = Input(b_data, :j, :k)
        c_matrix = sprand(Bool, 100, 100, .1)
        c_data = Tensor(SparseList(SparseList(Element(0), 100), 100))
        copyto!(c_data, c_matrix)
        c = Input(c_data, :k, :l)
        d = Aggregate(+, :j, MapJoin(*, a, b))
        e = Query(:out, Materialize(t_sparse_list, t_sparse_list, :i, :l, Aggregate(+, :k, MapJoin(*, d, c))))
        verbose= 0
        result = galley(e, verbose=verbose)
        d_matrix = a_matrix * b_matrix
        correct_matrix = d_matrix * c_matrix
        @test result.value == correct_matrix
    end

    @testset "100x100 matrices, diagonal mult" begin
        a_matrix = sprand(Bool, 100, 100, .1)
        a_data = Tensor(SparseList(SparseList(Element(0), 100), 100))
        copyto!(a_data, a_matrix)
        a = Input(a_data, :i, :i)
        b_matrix = sprand(Bool, 100, 100, .1)
        b_data = Tensor(SparseList(SparseList(Element(0), 100), 100))
        copyto!(b_data, b_matrix)
        b = Input(b_data, :i, :i)
        d = Query(:out, Materialize(t_dense, :i, MapJoin(*, a, b)))
        result = galley(d, verbose=verbose)
        correct_matrix = spzeros(100)
        for i in 1:100
            correct_matrix[i] = a_matrix[i,i] * b_matrix[i,i]
        end
        @test result.value == correct_matrix
    end
end

nothing
