b = TBLIS()

# test simple methods
#---------------------
typelist = (Float32, Float64, ComplexF32, ComplexF64)
@testset "simple methods with eltype = $T" for T in typelist
    @testset "tensorcopy" begin
        A = randn(T, (3, 5, 4, 6))
        p = (3, 1, 4, 2)
        C1 = permutedims(A, p)
        C2 = @inferred tensorcopy((p...,), A, (1:4...,); backend=b)
        C3 = @inferred tensorcopy(A, (p, ()), false, 1, b)
        @test C1 ≈ C2
        @test C2 == C3
        @test C1 ≈ ncon(Any[A], Any[[-2, -4, -1, -3]])
    end

    @testset "tensoradd" begin
        A = randn(T, (3, 5, 4, 6))
        B = randn(T, (5, 6, 3, 4))
        p = (3, 1, 4, 2)
        C1 = A + permutedims(B, p)
        C2 = @inferred tensoradd(A, p, B, (1:4...,); backend=b)
        C3 = @inferred tensoradd(A, ((1:4...,), ()), false, B, (p, ()), false, 1, 1, b)
        @test C1 ≈ C2
        @test C2 == C3
    end

    @testset "tensortrace" begin
        A = randn(Float64, (50, 100, 100))
        C1 = tensortrace(A, [:a, :b, :b])
        C2 = tensortrace(A, [:a, :b, :b]; backend=b)
        C3 = ncon(Any[A], Any[[-1, 1, 1]]; backend=b)
        @test C1 ≈ C2
        @test C2 == C3
        A = randn(Float64, (3, 20, 5, 3, 20, 4, 5))
        C1 = tensortrace((:e, :a, :d), A, (:a, :b, :c, :d, :b, :e, :c))
        C2 = @inferred tensortrace((:e, :a, :d), A, (:a, :b, :c, :d, :b, :e, :c); backend=b)
        C3 = @inferred tensortrace(A, ((6, 1, 4), ()), ((2, 3), (5, 7)), false, 1.0, b)
        C4 = ncon(Any[A], Any[[-2, 1, 2, -3, 1, -1, 2]]; backend=b)
        @test C1 ≈ C2
        @test C2 == C3 == C4
    end

    @testset "tensorcontract" begin
        A = randn(T, (3, 20, 5, 3, 4))
        B = randn(T, (5, 6, 20, 3))
        C1 = tensorcontract((:a, :g, :e, :d, :f), A, (:a, :b, :c, :d, :e), B,
                            (:c, :f, :b, :g))
        C2 = @inferred tensorcontract((:a, :g, :e, :d, :f),
                                      A, (:a, :b, :c, :d, :e), B, (:c, :f, :b, :g);
                                      backend=b)
        C3 = @inferred tensorcontract(A, ((1, 4, 5), (2, 3)), false, B, ((3, 1), (2, 4)),
                                      false, ((1, 5, 3, 2, 4), ()), 1, b)
        C4 = @inferred tensorcontract(A, ((1, 4, 5), (2, 3)), false, B, ((3, 1), (2, 4)),
                                      false, ((1, 5, 3, 2, 4), ()), 1, b,
                                      ManualAllocator())
        C5 = ncon(Any[A, B], Any[[-1, 1, 2, -4, -3], [2, -5, 1, -2]]; backend=b,
                  allocator=ManualAllocator())

        @test C1 ≈ C2
        @test C2 == C3 == C4 == C5
    end

    @testset "tensorproduct" begin
        A = randn(T, (5, 5, 5, 5))
        B = rand(T, (5, 5, 5, 5))
        C1 = kron(reshape(B, (25, 25)), reshape(A, (25, 25)))
        C2 = reshape((@inferred tensorproduct((1, 2, 5, 6, 3, 4, 7, 8),
                                              A, (1, 2, 3, 4), B, (5, 6, 7, 8); backend=b)),
                     (5 * 5 * 5 * 5, 5 * 5 * 5 * 5))
        @test C1 ≈ C2

        A = rand(1, 2)
        B = rand(4, 5)
        C1 = tensorcontract((-1, -2, -3, -4), A, (-3, -1), false, B, (-2, -4), false)
        C2 = tensorcontract((-1, -2, -3, -4), A, (-3, -1), false, B, (-2, -4), false;
                            backend=b)
        C3 = tensorproduct(A, ((1, 2), ()), false, B, ((), (1, 2)), false,
                           ((2, 3, 1, 4), ()), 1, b)
        C4 = tensorproduct(A, ((1, 2), ()), false, B, ((), (1, 2)), false,
                           ((2, 3, 1, 4), ()), 1, b, ManualAllocator())
        @test C1 ≈ C2
        @test C2 == C3 == C4
    end
end

# test in-place methods
#-----------------------
# test different versions of in-place methods,
# with changing element type and with nontrivial strides
@testset "in-place methods" begin
    @testset "tensorcopy!" begin
        Abig = randn(Float64, (30, 30, 30, 30))
        A = view(Abig, 1 .+ 3 * (0:9), 2 .+ 2 * (0:6), 5 .+ 3 * (0:6), 4 .+ 3 * (0:8))
        p = (3, 1, 4, 2)
        Cbig = zeros(Float64, (50, 50, 50, 50))
        C = view(Cbig, 13 .+ (0:6), 11 .+ 2 * (0:9), 7 .+ 5 * (0:8), 4 .+ 5 * (0:6))
        Acopy = tensorcopy(A, 1:4)
        Ccopy = tensorcopy(C, 1:4)
        pA = (p, ())
        α = randn(Float64)
        tensorcopy!(C, A, pA, false, α, b)
        tensorcopy!(Ccopy, Acopy, pA, false, 1.0, b)
        @test C ≈ α * Ccopy
        @test_throws IndexError tensorcopy!(C, A, ((1, 2, 3), ()), false, 1.0, b)
        @test_throws DimensionMismatch tensorcopy!(C, A, ((1, 2, 3, 4), ()), false, 1.0, b)
        @test_throws IndexError tensorcopy!(C, A, ((1, 2, 2, 3), ()), false, 1.0, b)
    end

    @testset "tensoradd!" begin
        Abig = randn(ComplexF32, (30, 30, 30, 30))
        A = view(Abig, 1 .+ 3 * (0:9), 2 .+ 2 * (0:6), 5 .+ 4 * (0:6), 4 .+ 3 * (0:8))
        p = (3, 1, 4, 2)
        Cbig = zeros(ComplexF32, (50, 50, 50, 50))
        C = view(Cbig, 13 .+ (0:6), 11 .+ 4 * (0:9), 15 .+ 4 * (0:8), 4 .+ 3 * (0:6))
        Ccopy = tensorcopy(1:4, C, 1:4)
        α = randn(ComplexF32)
        β = randn(ComplexF32)
        tensoradd!(C, A, (p, ()), false, α, β, b)
        tensoradd!(Ccopy, A, (p, ()), false, α, β) # default backend
        @test C ≈ Ccopy
        @test_throws IndexError tensoradd!(C, A, ((1, 2, 3), ()), false, 1.2, 0.5, b)
        @test_throws DimensionMismatch tensoradd!(C, A, ((1, 2, 3, 4), ()), false, 1.2, 0.5,
                                                  b)
        @test_throws IndexError tensoradd!(C, A, ((1, 1, 2, 3), ()), false, 1.2, 0.5, b)
    end

    @testset "tensortrace!" begin
        Abig = rand(ComplexF64, (30, 30, 30, 30))
        A = view(Abig, 1 .+ 3 * (0:8), 2 .+ 2 * (0:14), 5 .+ 4 * (0:6), 7 .+ 2 * (0:8))
        Bbig = rand(ComplexF64, (50, 50))
        B = view(Bbig, 13 .+ (0:14), 3 .+ 5 * (0:6))
        Acopy = tensorcopy(A, 1:4)
        Bcopy = tensorcopy(B, 1:2)
        α = randn(Float64)
        β = randn(Float64)
        tensortrace!(B, A, ((2, 3), ()), ((1,), (4,)), true, α, β, b)
        tensortrace!(Bcopy, A, ((2, 3), ()), ((1,), (4,)), true, α, β) # default backend
        @test B ≈ Bcopy
        @test_throws IndexError tensortrace!(B, A, ((1,), ()), ((2,), (3,)), false, α, β, b)
        @test_throws DimensionMismatch tensortrace!(B, A, ((1, 4), ()), ((2,), (3,)), false,
                                                    α, β, b)
        @test_throws IndexError tensortrace!(B, A, ((1, 4), ()), ((1, 1), (4,)), false, α,
                                             β, b)
        @test_throws IndexError tensortrace!(B, A, ((1, 4), ()), ((1,), (3,)), false,
                                             α, β, b)
    end

    bref = TensorOperations.DefaultBackend() # reference backend
    @testset "tensorcontract! with allocator = $allocator" for allocator in
                                                               (DefaultAllocator(),
                                                                ManualAllocator())
        Abig = rand(ComplexF64, (30, 30, 30, 30))
        A = view(Abig, 1 .+ 3 * (0:8), 2 .+ 2 * (0:14), 5 .+ 4 * (0:6), 7 .+ 2 * (0:8))
        Bbig = rand(ComplexF64, (50, 50, 50))
        B = view(Bbig, 3 .+ 5 * (0:6), 7 .+ 2 * (0:7), 13 .+ (0:14))
        Cbig = rand(ComplexF64, (40, 40, 40))
        C = view(Cbig, 3 .+ 2 * (0:8), 13 .+ (0:8), 7 .+ 3 * (0:7))
        Acopy = tensorcopy(A, 1:4)
        Bcopy = tensorcopy(B, 1:3)
        Ccopy = tensorcopy(C, 1:3)
        α = randn(ComplexF64)
        β = randn(ComplexF64)
        tensorcontract!(C, A, ((4, 1), (2, 3)), false, B, ((3, 1), (2,)), true,
                        ((1, 2, 3), ()), α, β, b, allocator)
        tensorcontract!(Ccopy, A, ((4, 1), (2, 3)), false, B, ((3, 1), (2,)), true,
                        ((1, 2, 3), ()), α, β, bref, allocator)
        @test C ≈ Ccopy

        Ccopy = tensorcopy(C, 1:3)
        tensorcontract!(C, A, ((4, 1), (2, 3)), true, B, ((3, 1), (2,)), true,
                        ((1, 2, 3), ()), α, β, b, allocator)
        tensorcontract!(Ccopy, A, ((4, 1), (2, 3)), true, B, ((3, 1), (2,)), true,
                        ((1, 2, 3), ()), α, β, bref, allocator)
        @test C ≈ Ccopy

        Ccopy = tensorcopy(C, 1:3)
        tensorcontract!(C, A, ((4, 1), (2, 3)), true, B, ((3, 1), (2,)), false,
                        ((1, 2, 3), ()), α, β, b, allocator)
        tensorcontract!(Ccopy, A, ((4, 1), (2, 3)), true, B, ((3, 1), (2,)), false,
                        ((1, 2, 3), ()), α, β, bref, allocator)
        @test C ≈ Ccopy

        @test_throws IndexError tensorcontract!(C,
                                                A, ((4, 1), (2, 4)), false,
                                                B, ((1, 3), (2,)), false,
                                                ((1, 2, 3), ()), α, β, b)
        @test_throws IndexError tensorcontract!(C,
                                                A, ((4, 1), (2, 3)), false,
                                                B, ((1, 3), ()), false,
                                                ((1, 2, 3), ()), α, β, b)
        @test_throws IndexError tensorcontract!(C,
                                                A, ((4, 1), (2, 3)), false,
                                                B, ((1, 3), (2,)), false,
                                                ((1, 2), ()), α, β, b)
        @test_throws DimensionMismatch tensorcontract!(C,
                                                       A, ((4, 1), (2, 3)), false,
                                                       B, ((1, 3), (2,)), false,
                                                       ((1, 3, 2), ()), α, β, b)
    end
end
