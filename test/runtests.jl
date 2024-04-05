using TensorOperations
using TensorOperationsTBLIS
using Test
using LinearAlgebra: norm

@testset "elementary operations" verbose = true begin
    @testset "tensorcopy" begin
        A = randn(Float32, (3, 5, 4, 6))
        @tensor C1[4, 1, 3, 2] := A[1, 2, 3, 4]
        @tensor backend = tblis C2[4, 1, 3, 2] := A[1, 2, 3, 4]
        @test C2 ≈ C1
    end

    @testset "tensoradd" begin
        A = randn(Float32, (5, 6, 3, 4))
        B = randn(Float32, (5, 6, 3, 4))
        α = randn(Float32)
        @tensor C1[a, b, c, d] := A[a, b, c, d] + α * B[a, b, c, d]
        @tensor backend = tblis C2[a, b, c, d] := A[a, b, c, d] + α * B[a, b, c, d]
        @test collect(C2) ≈ C1

        C = randn(ComplexF32, (5, 6, 3, 4))
        D = randn(ComplexF32, (5, 3, 4, 6))
        β = randn(ComplexF32)
        @tensor E1[a, b, c, d] := C[a, b, c, d] + β * conj(D[a, c, d, b])
        @tensor backend = tblis E2[a, b, c, d] := C[a, b, c, d] + β * conj(D[a, c, d, b])
        @test collect(E2) ≈ E1
    end

    @testset "tensortrace" begin
        A = randn(Float32, (5, 10, 10))
        @tensor B1[a] := A[a, b′, b′]
        @tensor backend = tblis B2[a] := A[a, b′, b′]
        @test B2 ≈ B1

        C = randn(ComplexF32, (3, 20, 5, 3, 20, 4, 5))
        @tensor D1[e, a, d] := C[a, b, c, d, b, e, c]
        @tensor backend = tblis D2[e, a, d] := C[a, b, c, d, b, e, c]
        @test D2 ≈ D1

        @tensor D3[a, e, d] := conj(C[a, b, c, d, b, e, c])
        @tensor backend = tblis D4[a, e, d] := conj(C[a, b, c, d, b, e, c])
        @test D4 ≈ D3

        α = randn(ComplexF32)
        @tensor D5[d, e, a] := α * C[a, b, c, d, b, e, c]
        @tensor backend = tblis D6[d, e, a] := α * C[a, b, c, d, b, e, c]
        @test D6 ≈ D5
    end

    @testset "tensorcontract" begin
        A = randn(Float32, (3, 20, 5, 3, 4))
        B = randn(Float32, (5, 6, 20, 3))
        @tensor C1[a, g, e, d, f] := A[a, b, c, d, e] * B[c, f, b, g]
        @tensor backend = tblis C2[a, g, e, d, f] := A[a, b, c, d, e] * B[c, f, b, g]
        @test C2 ≈ C1

        D = randn(ComplexF64, (3, 3, 3))
        E = rand(ComplexF64, (3, 3, 3))
        @tensor F1[a, b, c, d, e, f] := D[a, b, c] * conj(E[d, e, f])
        @tensor backend = tblis F2[a, b, c, d, e, f] := D[a, b, c] * conj(E[d, e, f])
        @test F2 ≈ F1 atol = 1e-12
    end
end

@testset "more complicated expressions" verbose = true begin
    Da, Db, Dc, Dd, De, Df, Dg, Dh = 10, 15, 4, 8, 6, 7, 3, 2
    A = rand(ComplexF64, (Dc, Da, Df, Da, De, Db, Db, Dg))
    B = rand(ComplexF64, (Dc, Dh, Dg, De, Dd))
    C = rand(ComplexF64, (Dd, Dh, Df))
    α = rand(ComplexF64)
    # α = 1

    @tensor D1[d, f, h] := A[c, a, f, a, e, b, b, g] * B[c, h, g, e, d] + α * C[d, h, f]
    @tensor backend = tblis D2[d, f, h] := A[c, a, f, a, e, b, b, g] * B[c, h, g, e, d] +
                                           α * C[d, h, f]
    @test D2 ≈ D1 rtol = 1e-8

    @test norm(vec(D1)) ≈ sqrt(abs(@tensor D1[d, f, h] * conj(D1[d, f, h])))
    @test norm(D2) ≈ sqrt(abs(@tensor backend = tblis D2[d, f, h] * conj(D2[d, f, h])))

    @testset "readme example" begin
        α = randn()
        A = randn(5, 5, 5, 5, 5, 5)
        B = randn(5, 5, 5)
        C = randn(5, 5, 5)
        D = zeros(5, 5, 5)
        D2 = copy(D)
        @tensor begin
            D[a, b, c] = A[a, e, f, c, f, g] * B[g, b, e] + α * C[c, a, b]
            E[a, b, c] := A[a, e, f, c, f, g] * B[g, b, e] + α * C[c, a, b]
        end
        @tensor backend = tblis begin
            D2[a, b, c] = A[a, e, f, c, f, g] * B[g, b, e] + α * C[c, a, b]
            E2[a, b, c] := A[a, e, f, c, f, g] * B[g, b, e] + α * C[c, a, b]
        end
        @test D2 ≈ D
        @test E2 ≈ E
    end

    @testset "tensor network examples ($T)" for T in
                                                (Float32, Float64, ComplexF32, ComplexF64)
        D1, D2, D3 = 30, 40, 20
        d1, d2 = 2, 3

        A1 = randn(T, D1, d1, D2)
        A2 = randn(T, D2, d2, D3)
        ρₗ = randn(T, D1, D1)
        ρᵣ = randn(T, D3, D3)
        H = randn(T, d1, d2, d1, d2)

        @tensor begin
            HrA12[a, s1, s2, c] := ρₗ[a, a'] * A1[a', t1, b] * A2[b, t2, c'] * ρᵣ[c', c] *
                                   H[s1, s2, t1, t2]
        end
        @tensor backend = tblis begin
            HrA12′[a, s1, s2, c] := ρₗ[a, a'] * A1[a', t1, b] * A2[b, t2, c'] * ρᵣ[c', c] *
                                    H[s1, s2, t1, t2]
        end
        @test HrA12′ ≈ HrA12

        @tensor begin
            E1 = ρₗ[a', a] * A1[a, s, b] * A2[b, s', c] * ρᵣ[c, c'] * H[t, t', s, s'] *
                 conj(A1[a', t, b']) * conj(A2[b', t', c'])
        end
        @tensor backend = tblis begin
            E2 = ρₗ[a', a] * A1[a, s, b] * A2[b, s', c] * ρᵣ[c, c'] * H[t, t', s, s'] *
                 conj(A1[a', t, b']) * conj(A2[b', t', c'])
        end
        @test E2 ≈ E1
    end
end
