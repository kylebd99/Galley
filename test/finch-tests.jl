
@testset "interface" begin

    Finch.set_scheduler!(Finch.LogicExecutor(GalleyOptimizer()))

    @info "Testing Finch Interface"

    #https://github.com/finch-tensor/Finch.jl/issues/383
    let
        A = [0.0 0.0 4.4; 1.1 0.0 0.0; 2.2 0.0 5.5; 3.3 0.0 0.0]
        A_fbr = Tensor(Dense(Dense(Element(0.0))), A)

        -A # works
        -A_fbr # used to fail
    end

    #https://github.com/finch-tensor/Finch.jl/issues/592
    let
        @test eltype(broadcast(Finch.fld_nothrow, Tensor(ones(Int16, 1)), Tensor(ones(Int8, 1)))) == Int16
        @test eltype(broadcast(Finch.fld_nothrow, Tensor(ones(Int16, 0)), Tensor(ones(Int8, 0)))) == Int16

        @test eltype(broadcast(Finch.fld_nothrow, Tensor(ones(Float64, 1)), Tensor(ones(Float32, 1)))) == Float64
        @test eltype(broadcast(Finch.fld_nothrow, Tensor(ones(Float64, 0)), Tensor(ones(Float32, 0)))) == Float64
    end

    #https://github.com/finch-tensor/Finch.jl/issues/578
    let
        @test maximum(Tensor(ones(Int8, 4,4))) === Int8(1)
        @test minimum(Tensor(ones(Int8, 4,4))) === Int8(1)
        @test extrema(Tensor(ones(Int8, 4,4))) === (Int8(1), Int8(1))
        @test compute(maximum(lazy(Tensor(ones(Int8, 4,4)))))[] === Int8(1)
        @test compute(minimum(lazy(Tensor(ones(Int8, 4,4)))))[] === Int8(1)
        @test compute(extrema(lazy(Tensor(ones(Int8, 4,4)))))[] === (Int8(1), Int8(1))
    end

    #https://github.com/finch-tensor/Finch.jl/issues/576
    let
        a = zeros(ComplexF64, 2, 1)
        a[1, 1] = 1.8 + 1.8im
        a[2, 1] = 4.8 + 4.8im

        b = [1 + 1im, 2 + 2im]

        a_tns = Tensor(a)
        b_tns = Tensor(b)

        res = Finch.tensordot(a_tns, b_tns, ((1,), (1,)))

        @test eltype(res) == ComplexF64
    end

    #https://github.com/finch-tensor/Finch.jl/issues/577
    let
        a = Tensor(ones(UInt8, 1))
        b = Tensor(ones(UInt8, 1))
        @test eltype(Finch.tensordot(a, b, 0)) == UInt8
    end

    for scheduler in [Finch.LogicExecutor(GalleyOptimizer())]
        Finch.with_scheduler(scheduler) do
            @info "Testing $scheduler"

            #https://github.com/finch-tensor/Finch.jl/issues/474
            let
                arr = [1 2 1 2; 2 1 2 1]
                arr2 = arr .+ 2

                tns = Tensor(Dense(Dense(Element(0))), arr)
                tns2 = Tensor(Dense(Dense(Element(0))), arr2)


                broadcast(/, tns, tns2)  # passes
                broadcast(Finch.fld_nothrow, tns, tns2)  # fails with RewriteTools.RuleRewriteError
                broadcast(Finch.rem_nothrow, tns, tns2)
            end

            #https://github.com/finch-tensor/Finch.jl/issues/520
            let
                A = rand(2, 2)
                x = rand(2)
                lx = lazy(x)
                y = compute(@einsum y[i] += A[i, j] * lx[j])
                @test norm(y .- A * x) < 1e-10
            end

            #https://github.com/finch-tensor/Finch.jl/issues/554
            let
                @test broadcast(trunc, swizzle(Tensor(ones(1)), 1)) == Tensor(ones(1))

                @test broadcast(trunc, swizzle(Tensor(ones(2)), 1)) == Tensor(ones(2))
            end

            #https://github.com/finch-tensor/Finch.jl/issues/533
            let
                A = lazy(fsprand(1, 1, 0.5))
                compute(sum(A .+ A)) #should not error
            end

            #https://github.com/finch-tensor/Finch.jl/issues/535
            let
                LEN = 10;
                a_raw = rand(Int, LEN, LEN - 5) * 10;
                b_raw = rand(Int, LEN, LEN - 5) * 10;
                c_raw = rand(Int, LEN, LEN) * 10;

                a = lazy(swizzle(Tensor(a_raw), 1, 2));
                b = lazy(swizzle(Tensor(b_raw), 1, 2));
                c = lazy(swizzle(Tensor(c_raw), 1, 2));

                ref = reshape(c_raw, 10, 10, 1) .* reshape(a_raw, 10, 1, 5) .* reshape(b_raw, 1, 10, 5);

                plan = c[:, :, nothing] .* a[:, nothing, :] .* b[nothing, :, :];
                @test compute(plan) == ref

                plan = broadcast(*, broadcast(*, c[:, :, nothing], a[:, nothing, :]), b[nothing, :, :]);
                @test compute(plan) == ref
            end

            #https://github.com/finch-tensor/Finch.jl/issues/536
            let
                A = [1 2; 3 4]
                swizzle(lazy(A), 2, 1) == permutedims(A)
            end

            #https://github.com/finch-tensor/Finch.jl/issues/530
            let
                A_tns = Tensor(Dense(Dense(Dense(Element(0.0)))), zeros(3, 3, 3))
                A_sw = swizzle(A_tns, 2, 3, 1)
                A_tns == A_sw #fails
            end

            #https://github.com/finch-tensor/Finch.jl/issues/524
            let
                arr3d = rand(Int, 3, 2, 3) .% 10
                tns = Tensor(Dense(Dense(Dense(Element(0)))), arr3d)

                tns_l = lazy(tns)
                reduced = sum(tns_l, dims=(1, 2))

                plan = broadcast(+, tns_l, reduced)
                result = compute(plan)
            end

            #https://github.com/finch-tensor/Finch.jl/issues/527
            let
                tns_1 = swizzle(Tensor(ones(10, 10)), 1, 2)
                tns_1[:, :] # == tns_1 https://github.com/finch-tensor/Finch.jl/issues/530

                tns_2 = swizzle(Tensor(ones(10)), 1)
                tns_2[:]# == tns_2 https://github.com/finch-tensor/Finch.jl/issues/530
            end

            #https://github.com/finch-tensor/Finch.jl/issues/528
            let
                tns = swizzle(Tensor(ones(10, 10)), 1, 2)
                @test tns[:, :] == ones(10, 10)
                @test tns[nothing, :, :] == ones(1, 10, 10)
                @test tns[:, nothing, :] == ones(10, 1, 10)
                @test tns[:, :, nothing] == ones(10, 10, 1)
            end

            #https://github.com/finch-tensor/Finch.jl/issues/428
            let
                @testset "Verbose" begin
                    a = [1 2; 3 4]
                    b = [5 6; 7 8]
                    a_l = lazy(a)
                    b_l = lazy(b)

                    c = permutedims(broadcast(.+, permutedims(a_l, (2, 1)), permutedims(b_l, (2, 1))), (2, 1))
                    compute(c, verbose=true)
                end
            end

            let

                @testset "Einsum Tests" begin
                    # Test 0
                    A = [1 2; 3 4]
                    B = [5 6; 7 8]
                    s = Scalar(0)
                    @einsum s[] += abs(A[i, k] * B[k, j])
                    @test s[] == 134

                    # Test 1
                    A = [1 2; 3 4]
                    B = [5 6; 7 8]
                    @einsum C[i, j] += A[i, k] * B[k, j]
                    @test C == [19 22; 43 50]

                    # Test 2
                    A = Tensor(Dense(SparseList(Element(0))), fsprand(Int, 3, 5, 0.5))
                    B = Tensor(Dense(SparseList(Element(0))), fsprand(Int, 5, 3, 0.5))
                    @einsum C[i, j, k] += A[i, j] * B[j, k]

                    C_ref = zeros(Int, 3, 5, 3)
                    for i = 1:3, j = 1:5, k = 1:3
                        C_ref[i, j, k] += A[i, j] * B[j, k]
                    end
                    @test C == C_ref

                    # Test 3
                    X = Tensor(Dense(SparseList(Element(0))), fsprand(Int, 4, 6, 0.5))
                    Y = Tensor(Dense(SparseList(Element(0))), fsprand(Int, 6, 4, 0.5))
                    @einsum D[i, k] += X[i, j] * Y[j, k]

                    D_ref = zeros(Int, 4, 4)
                    for i = 1:4, j = 1:6, k = 1:4
                        D_ref[i, k] += X[i, j] * Y[j, k]
                    end
                    @test D == D_ref

                    # Test 4
                    H = Tensor(Dense(SparseList(Element(0))), fsprand(Int, 5, 5, 0.6))
                    I = Tensor(Dense(SparseList(Element(0))), fsprand(Int, 5, 5, 0.6))
                    @einsum J[i, j] = H[i, j] * I[i, j]

                    J_ref = zeros(Int, 5, 5)
                    for i = 1:5, j = 1:5
                        J_ref[i, j] = H[i, j] * I[i, j]
                    end
                    @test J == J_ref

                    # Test 5
                    K = Tensor(Dense(SparseList(Element(0))), fsprand(Int, 4, 4, 0.7))
                    L = Tensor(Dense(SparseList(Element(0))), fsprand(Int, 4, 4, 0.7))
                    M = Tensor(Dense(SparseList(Element(0))), fsprand(Int, 4, 4, 0.7))
                    @einsum N[i, j] += K[i, k] * L[k, j] - M[i, j]

                    N_ref = zeros(Int, 4, 4)
                    for i = 1:4, k = 1:4, j = 1:4
                        N_ref[i, j] += K[i, k] * L[k, j] - M[i, j]
                    end
                    @test N == N_ref

                    # Test 6
                    P = Tensor(Dense(SparseList(Element(-Inf))), fsprand(Int, 3, 3, 0.7)) # Adjacency matrix with probabilities
                    Q = Tensor(Dense(SparseList(Element(-Inf))), fsprand(Int, 3, 3, 0.7))
                    @einsum init=-Inf R[i, j] <<max>>= P[i, k] + Q[k, j]  # Max-plus product

                    R_ref = fill(-Inf, 3, 3)
                    for i = 1:3, j = 1:3
                        for k = 1:3
                            R_ref[i, j] = max(R_ref[i, j], P[i, k] + Q[k, j])
                        end
                    end
                    @test R == R_ref

                    # Test for Sparse Matrix-Vector Multiplication (SpMV)
                    # Define a sparse matrix `S` and a dense vector `v`
                    S = Tensor(Dense(SparseList(Element(0))), fsprand(Int, 10, 10, 0.3))  # 10x10 sparse matrix with 30% density
                    v = Tensor(Dense(Element(0)), rand(Int, 10))              # Dense vector of size 10

                    # Perform matrix-vector multiplication using the @einsum macro
                    @einsum w[i] += S[i, k] * v[k]  # Compute the product

                    # Reference calculation using explicit loop for validation
                    w_ref = zeros(Int, 10)
                    for i = 1:10
                        for k = 1:10
                            w_ref[i] += S[i, k] * v[k]
                        end
                    end

                    # Test to ensure the results match
                    @test w == w_ref

                    # Test for Transposed Sparse Matrix-Vector Multiplication (SpMV)
                    # Define a sparse matrix `T` and a dense vector `u`
                    T = Tensor(Dense(SparseList(Element(0))), fsprand(Int, 10, 10, 0.3))  # 10x10 sparse matrix with 30% density
                    u = Tensor(Dense(Element(0)), rand(Int, 10))              # Dense vector of size 10

                    # Perform transposed matrix-vector multiplication using the @einsum macro
                    @einsum x[k] += T[j, k] * u[j]  # Compute the product using the transpose of T

                    # Reference calculation using explicit loop for validation
                    x_ref = zeros(Int, 10)
                    for k = 1:10
                        for j = 1:10
                            x_ref[k] += T[j, k] * u[j]
                        end
                    end

                    # Test to ensure the results match
                    @test x == x_ref

                    # Test for Outer Product with Output Named A
                    # Define two vectors for outer product
                    v1 = Tensor(Dense(Element(0)), rand(Int, 5))  # Dense vector of size 5
                    v2 = Tensor(Dense(Element(0)), rand(Int, 7))  # Dense vector of size 7

                    # Perform outer product using the @einsum macro
                    @einsum A[i, j] = v1[i] * v2[j]  # Compute the outer product

                    # Reference calculation using explicit loop for validation
                    A_ref = zeros(Int, 5, 7)
                    for i = 1:5
                        for j = 1:7
                            A_ref[i, j] = v1[i] * v2[j]
                        end
                    end

                    # Test to ensure the results match
                    @test A == A_ref


                    # Test for multiplying a vector by a Scalar
                    v = Tensor(Dense(Element(0)), rand(Int, 5))
                    n = 7

                    #Perform scalar multiplcation
                    @einsum A[i] = n*v[i]

                    # Reference Calculation using explicit loop for validation
                    A_ref = Tensor(Dense(Element(0)), rand(Int, 5))
                    for i = 1:5
                        A_ref[i] = v[i]*n
                    end

                    #Test to ensure the results match
                    @test A == A_ref


                end
            end

            A = Tensor(SparseList(Element(0.0)), fsparse([1, 3, 5, 7, 9], [2.0, 3.0, 4.0, 5.0, 6.0], (10,)))
            B = Tensor(SparseList(Element(0.0)), A)
            @test A == B

            A = [0.0 0.0 0.0 0.0; 1.0 0.0 0.0 1.0]
            B = Tensor(Dense(SparseList(Element(0.0))), A)
            C = Tensor(Dense(Dense(Element(0.0))), A)
            @test A == B

            A = [0 0; 0 0]
            B = Tensor(Dense(Dense(Element(0.0))), A)
            @test A == B

            A = Tensor(Dense(Element(0.0)), [0, 0, 0, 0])
            B = Tensor(Dense(Element(0.0)), [0, 0, 0, 0, 0])
            @test size(A) != size(B) && A != B

            A = [0 0 0 0 1 0 0 1]
            B = Tensor(Dense(SparseList(Element(0))), [0 0 0 0; 1 0 0 1])
            @test size(A) != size(B) && A != B

            A = Tensor(Dense(SparseList(Element(0.0))), [1 0 0 0; 1 1 0 0; 1 1 1 0])
            B = [0 0 0 0; 1 1 0 0; 1 1 1 0]
            @test size(A) == size(B) && A != B
            C = Tensor(Dense(SparseList(Element(0.0))), [0 0 0 0; 1 1 0 0; 1 1 1 0])
            @test B == C

            A = [NaN, 0.0, 3.14, 0.0]
            B = Tensor(SparseList(Element(0.0)), [NaN, 0.0, 3.14, 0.0])
            C = Tensor(SparseList(Element(0.0)), [NaN, 0.0, 3.14, 0.0])
            D = [1.0, 2.0, 4.0, 8.0]
            @test isequal(A, B)
            @test isequal(A, C)
            @test isequal(B, C)
            @test isequal(B, A)
            @test !isequal(A, D)
            @test A != B

            let
                A_ref = [0.0 0.0 4.4; 1.1 0.0 0.0; 2.2 0.0 5.5; 3.3 0.0 0.0]
                A_ref = A_ref * floatmax()/sum(A_ref)
                A = Tensor(Dense(SparseList(Element(0.0))), A_ref)

                @test sum(A) == sum(A_ref)
                @test minimum(A) == minimum(A_ref)
                @test maximum(A) == maximum(A_ref)
                @test extrema(A) == extrema(A_ref)
                @test norm(A) == norm(A_ref)
                @test norm(A, -Inf) == norm(A_ref, -Inf)
                @test norm(A, 0) == norm(A_ref, 0)
                @test norm(A, 1) == norm(A_ref, 1)
                @test norm(A, 1.5) == norm(A_ref, 1.5)
                @test norm(A, Inf) == norm(A_ref, Inf)
            end

            let
                A = Tensor(Dense(SparseList(Element(0.0))), [0.0 0.0 4.4; 1.1 0.0 0.0; 2.2 0.0 5.5; 3.3 0.0 0.0])
                B = Tensor(Dense(SparseList(Element(0.0))), [0.0 0.0 4.4; 1.1 0.0 0.0; 2.2 0.0 5.5; 3.3 0.0 0.0])
                C = lazy(A)
                D = lazy(B)
                E = (C + D) * 0.5
                F = compute(E)
                @test F == A
            end

            let
                A = Tensor(Dense(SparseList(Element(0))), [0 0 44; 11 0 0; 22 00 55; 33 0 0])
                B = Tensor(Dense(SparseList(Element(0))), [0 0 44; 11 0 0; 22 00 55; 33 0 0])
                c_correct = Tensor(Dense(Dense(Element(0))), [1936 0 2420 0; 0 121 242 363; 2420 242 3509 726; 0 363 726 1089])
                c = compute(tensordot(lazy(A), lazy(B), ((2, ), (2,)), init=0))
                @test c == c_correct
            end

            let
                A = lazy(Tensor(Dense(SparseList(Element(0))), [0 0 44; 11 0 0; 22 00 55; 33 0 0]))
                B = lazy(Tensor(Dense(SparseList(Element(0))), [0 0 44; 11 0 0; 22 00 55; 33 0 0]'))
                c_correct = Tensor(Dense(Dense(Element(0))), [1936 0 2420 0; 0 121 242 363; 2420 242 3509 726; 0 363 726 1089])
                c = compute(sum(A[:, :, nothing] .* B[nothing, :, :], dims=[2]))
                @test c == c_correct
            end

            #https://github.com/finch-tensor/Finch.jl/issues/457
            let
                A = zeros(2, 3, 3)
                A[1, :, :] = [1 2 3; 4 5 6; 7 8 9]
                A[2, :, :] = [1 1 1; 2 2 2; 3 3 3]
                perm = (2, 3, 1)
                A_t = permutedims(A, perm)

                A_tns = Tensor(Dense(Dense(Dense(Element(0.0)))), A)
                A_sw = swizzle(A_tns, perm...)
                A_lazy = lazy(A_sw)

                A_result = compute(A_lazy)

                @test Array(A_result) == A_t
                @test permutedims(A_tns, perm) == A_t
            end

            #https://github.com/finch-tensor/Finch.jl/pull/477
            let
                A = zeros(2, 3, 3)
                A_tns = Tensor(Dense(Dense(Dense(Element(0.0)))), A)

                @test compute(A) == A #If the scheduler improves, we can change this to ===
                @test compute(A_tns) == A_tns #If the scheduler improves, we can change this to ===
            end

            #https://github.com/finch-tensor/Finch.jl/issues/481
            let
                r = fsprand(1, 10, 10, 0.01)
                r_tns = Tensor(Dense(Dense(Dense(Element(0.0)))), r)
                @test r_tns + r_tns == 2 * r_tns
            end

            #https://github.com/finch-tensor/Finch.jl/issues/487
            let
                a = fsprand(10, 1, 0.8)
                b = fsprand(10, 1, 0.8)

                permutedims(broadcast(+, permutedims(a, (2, 1)), permutedims(b, (2, 1))), (2, 1))  # passes

                a_l = lazy(a)
                b_l = lazy(b)

                plan = permutedims(broadcast(+, permutedims(a_l, (2, 1)), permutedims(b_l, (2, 1))), (2, 1))
                compute(plan)  # fails
            end
        end
    end
end
