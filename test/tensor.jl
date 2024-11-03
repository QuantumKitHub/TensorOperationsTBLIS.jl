# test index notation using @tensor macro
#-----------------------------------------
b = TBLIS()

@testset "tensorcontract 1" begin
    A = randn(Float64, (3, 5, 4, 6))
    p = (4, 1, 3, 2)
    C1 = permutedims(A, p)
    @tensor backend = b C2[4, 1, 3, 2] := A[1, 2, 3, 4]
    @test C1 ≈ C2

    B = randn(Float64, (5, 6, 3, 4))
    p = [3, 1, 4, 2]
    @tensor backend = b C1[3, 1, 4, 2] := A[3, 1, 4, 2] + B[1, 2, 3, 4]
    C2 = A + permutedims(B, p)
    @test C1 ≈ C2
    @test_throws DimensionMismatch begin
        @tensor backend = b C[1, 2, 3, 4] := A[1, 2, 3, 4] + B[1, 2, 3, 4]
    end

    A = randn(Float64, (50, 100, 100))
    @tensor backend = b C1[a] := A[a, b', b']
    @tensor C2[a] := A[a, b', b']
    @test C1 ≈ C2

    A = randn(Float64, (3, 20, 5, 3, 20, 4, 5))
    @tensor backend = b C1[e, a, d] := A[a, b, c, d, b, e, c]
    @tensor C2[e, a, d] := A[a, b, c, d, b, e, c]
    @test C1 ≈ C2

    A = randn(Float64, (3, 20, 5, 3, 4))
    B = randn(Float64, (5, 6, 20, 3))
    @tensor backend = b C1[a, g, e, d, f] := A[a, b, c, d, e] * B[c, f, b, g]
    @tensor C2[a, g, e, d, f] := A[a, b, c, d, e] * B[c, f, b, g]
    @test C1 ≈ C2
end

@testset "tensorcontract 2" begin
    A = randn(ComplexF32, (5, 5, 5, 5))
    B = rand(ComplexF32, (5, 5, 5, 5))
    @tensor backend = b C1[1, 2, 5, 6, 3, 4, 7, 8] := A[1, 2, 3, 4] * B[5, 6, 7, 8]
    @tensor C2[1, 2, 5, 6, 3, 4, 7, 8] := A[1, 2, 3, 4] * B[5, 6, 7, 8]
    @test C1 ≈ C2
    @test_throws IndexError begin
        @tensor backend = b C[a, b, c, d, e, f, g, i] := A[a, b, c, d] * B[e, f, g, h]
    end
end

@testset "tensorcontract 3" begin
    Da, Db, Dc, Dd, De, Df, Dg, Dh = 10, 15, 4, 8, 6, 7, 3, 2
    A = rand(ComplexF64, (Da, Dc, Df, Da, De, Db, Db, Dg))
    B = rand(ComplexF64, (Dc, Dh, Dg, De, Dd))
    C = rand(ComplexF64, (Dd, Dh, Df))
    @tensor backend = b D1[d, f, h] := A[a, c, f, a, e, b, b, g] * B[c, h, g, e, d] +
                                       0.5 * C[d, h, f]
    @tensor D2[d, f, h] := A[a, c, f, a, e, b, b, g] * B[c, h, g, e, d] + 0.5 * C[d, h, f]
    @test D1 ≈ D2
    @test norm(vec(D1)) ≈
          sqrt(abs((@tensor backend = b tensorscalar(D1[d, f, h] * conj(D1[d, f, h])))))
end

@testset "views" begin
    p = [3, 1, 4, 2]
    Abig = randn(Float32, (30, 30, 30, 30))
    A = view(Abig, 1 .+ 3 .* (0:9), 2 .+ 2 .* (0:6), 5 .+ 4 .* (0:6), 4 .+ 3 .* (0:8))
    Cbig = zeros(Float32, (50, 50, 50, 50))
    C = view(Cbig, 13 .+ (0:6), 11 .+ 4 .* (0:9), 15 .+ 4 .* (0:8), 4 .+ 3 .* (0:6))
    Ccopy = copy(C)
    @tensor backend = b C[3, 1, 4, 2] = A[1, 2, 3, 4]
    @tensor Ccopy[3, 1, 4, 2] = A[1, 2, 3, 4]
    @test C ≈ Ccopy
end

@testset "views 2" begin
    p = [3, 1, 4, 2]
    Abig = randn(ComplexF64, (30, 30, 30, 30))
    A = view(Abig, 1 .+ 3 .* (0:9), 2 .+ 2 .* (0:6), 5 .+ 4 .* (0:6), 4 .+ 3 .* (0:8))
    Cbig = zeros(ComplexF64, (50, 50, 50, 50))
    C = view(Cbig, 13 .+ (0:6), 11 .+ 4 .* (0:9), 15 .+ 4 .* (0:8), 4 .+ 3 .* (0:6))
    α = randn(Float64)
    β = randn(Float64)
    @tensor backend = b D[3, 1, 4, 2] := β * C[3, 1, 4, 2] + α * A[1, 2, 3, 4]
    @tensor Dcopy[3, 1, 4, 2] := β * C[3, 1, 4, 2] + α * A[1, 2, 3, 4]
    @test D ≈ Dcopy
end

@testset "views 3" begin
    Abig = rand(Float64, (30, 30, 30, 30))
    A = view(Abig, 1 .+ 3 .* (0:8), 2 .+ 2 .* (0:14), 5 .+ 4 .* (0:6), 7 .+ 2 .* (0:8))
    Bbig = rand(Float64, (50, 50))
    B = view(Bbig, 13 .+ (0:14), 3 .+ 5 .* (0:6))
    Bcopy = copy(B)
    α = randn(Float64)
    @tensor backend = b B[b, c] += α * A[a, b, c, a]
    @tensor Bcopy[b, c] += α * A[a, b, c, a]
    @test B ≈ Bcopy
end

@testset "views 4" begin
    Abig = rand(ComplexF64, (30, 30, 30, 30))
    A = view(Abig, 1 .+ 3 .* (0:8), 2 .+ 2 .* (0:14), 5 .+ 4 .* (0:6), 7 .+ 2 .* (0:8))
    Bbig = rand(ComplexF64, (50, 50, 50))
    B = view(Bbig, 3 .+ 5 .* (0:6), 7 .+ 2 .* (0:7), 13 .+ (0:14))
    Cbig = rand(ComplexF64, (40, 40, 40))
    C = view(Cbig, 3 .+ 2 .* (0:8), 13 .+ (0:8), 7 .+ 3 .* (0:7))
    Ccopy = copy(C)
    α = randn(Float64)
    @tensor backend = b C[d, a, e] -= α * A[a, b, c, d] * conj(B[c, e, b])
    @tensor Ccopy[d, a, e] -= α * A[a, b, c, d] * conj(B[c, e, b])
    @test C ≈ Ccopy
end

@testset "Float32 views" begin
    α = randn(Float64)
    Abig = rand(ComplexF32, (30, 30, 30, 30))
    A = view(Abig, 1 .+ 3 .* (0:8), 2 .+ 2 .* (0:14), 5 .+ 4 .* (0:6), 7 .+ 2 .* (0:8))
    Bbig = rand(ComplexF32, (50, 50, 50))
    B = view(Bbig, 3 .+ 5 .* (0:6), 7 .+ 2 .* (0:7), 13 .+ (0:14))
    Cbig = rand(ComplexF32, (40, 40, 40))
    C = view(Cbig, 3 .+ 2 .* (0:8), 13 .+ (0:8), 7 .+ 3 .* (0:7))
    Ccopy = copy(C)
    @tensor backend = b C[d, a, e] += α * A[a, b, c, d] * conj(B[c, e, b])
    @tensor Ccopy[d, a, e] += α * A[a, b, c, d] * conj(B[c, e, b])
    @test C ≈ Ccopy
end

# Simple function example
# @tensor function f(A, b)
#     w[x] := (1 // 2) * A[x, y] * b[y]
#     return w
# end
# for T in (Float32, Float64, ComplexF32, ComplexF64, BigFloat)
#     A = rand(T, 10, 10)
#     b = rand(T, 10)
#     @test f(A, b) ≈ (1 // 2) * A * b
# end

# Example from README.md
@testset "README example" begin
    α = randn()
    A = randn(5, 5, 5, 5, 5, 5)
    B = randn(5, 5, 5)
    C = randn(5, 5, 5)
    D = zeros(5, 5, 5)
    @tensor backend = b begin
        D[a, b, c] = A[a, e, f, c, f, g] * B[g, b, e] + α * C[c, a, b]
        E[a, b, c] := A[a, e, f, c, f, g] * B[g, b, e] + α * C[c, a, b]
    end
    @test D == E
end

# Some tensor network examples
scalartypelist = (Float32, Float64, ComplexF32, ComplexF64)
@testset "tensor network examples ($T)" for T in scalartypelist
    D1, D2, D3 = 30, 40, 20
    d1, d2 = 2, 3
    A1 = rand(T, D1, d1, D2) .- 1 // 2
    A2 = rand(T, D2, d2, D3) .- 1 // 2
    rhoL = rand(T, D1, D1) .- 1 // 2
    rhoR = rand(T, D3, D3) .- 1 // 2
    H = rand(T, d1, d2, d1, d2) .- 1 // 2
    A12 = reshape(reshape(A1, D1 * d1, D2) * reshape(A2, D2, d2 * D3), (D1, d1, d2, D3))
    @tensor HrA12[a, s1, s2, c] := rhoL[a, a'] * A1[a', t1, b] * A2[b, t2, c'] *
                                   rhoR[c', c] * H[s1, s2, t1, t2]
    E = dot(A12, HrA12)
    @tensor backend = b HrA12′[a, s1, s2, c] := rhoL[a, a'] * A1[a', t1, b] *
                                                A2[b, t2, c'] *
                                                rhoR[c', c] * H[s1, s2, t1, t2]
    @tensor backend = b HrA12′′[:] := rhoL[-1, 1] * H[-2, -3, 4, 5] * A2[2, 5, 3] *
                                      rhoR[3, -4] *
                                      A1[1, 4, 2] # should be contracted in exactly same order
    @tensor backend = b order = (a', b, c', t1, t2) begin
        HrA12′′′[a, s1, s2, c] := rhoL[a, a'] * H[s1, s2, t1, t2] * A2[b, t2, c'] *
                                  rhoR[c', c] * A1[a', t1, b] # should be contracted in exactly same order
    end
    @tensor backend = b opt = true HrA12′′′′[:] := rhoL[-1, 1] * H[-2, -3, 4, 5] *
                                                   A2[2, 5, 3] *
                                                   rhoR[3, -4] * A1[1, 4, 2]

    @test HrA12′ == HrA12′′ == HrA12′′′ # should be exactly equal
    @test HrA12 ≈ HrA12′
    @test HrA12 ≈ HrA12′′′′
    @test HrA12′′ ≈ ncon([rhoL, H, A2, rhoR, A1],
                         [[-1, 1], [-2, -3, 4, 5], [2, 5, 3], [3, -4], [1, 4, 2]]; backend=b)
    @test HrA12′′ == @ncon([rhoL, H, A2, rhoR, A1],
                           [[-1, 1], [-2, -3, 4, 5], [2, 5, 3], [3, -4], [1, 4, 2]];
                           order=[1, 2, 3, 4, 5], output=[-1, -2, -3, -4], backend=b)
    @test E ≈
          @tensor backend = b tensorscalar(rhoL[a', a] * A1[a, s, b] * A2[b, s', c] *
                                           rhoR[c, c'] *
                                           H[t, t', s, s'] * conj(A1[a', t, b']) *
                                           conj(A2[b', t', c']))
    @test E ≈ @ncon([rhoL, A1, A2, rhoR, H, conj(A1), conj(A2)],
                    [[5, 1], [1, 2, 3], [3, 4, 9], [9, 10], [6, 8, 2, 4], [5, 6, 7],
                     [7, 8, 10]]; backend=b)
    # this implicitly tests that `ncon` returns a scalar if no open indices
end
