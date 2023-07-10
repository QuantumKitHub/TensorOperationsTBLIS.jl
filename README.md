# TensorOperationsTBLIS.jl
[tblis](https://github.com/devinamatthews/tblis) wrapper for [TensorOperations.jl]()

[![CI][ci-img]][ci-url] [![CI (Julia nightly)][ci-julia-nightly-img]][ci-julia-nightly-url] [![][codecov-img]][codecov-url]

[ci-img]: https://github.com/lkdvos/TensorOperationsTBLIS.jl/actions/workflows/ci.yml/badge.svg
[ci-url]: https://github.com/lkdvos/TensorOperationsTBLIS.jl/actions/workflows/ci.yml

[ci-julia-nightly-img]: https://github.com/lkdvos/TensorOperationsTBLIS.jl/actions/workflows/ci-julia-nightly.yml/badge.svg
[ci-julia-nightly-url]: https://github.com/lkdvos/TensorOperationsTBLIS.jl/actions/workflows/ci-julia-nightly.yml

[codecov-img]: https://codecov.io/gh/lkdvos/TensorOperationsTBLIS.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/lkdvos/TensorOperationsTBLIS.jl

Currently provides implementations of `tensorcontract!` and `tensoradd!` for `StridedArray{<:BlasFloat}`. These can be accessed through the backend system of TensorOperations, i.e.
```julia
using TensorOperations
using TensorOperationsTBLIS

α = randn()
A = randn(5, 5, 5, 5, 5, 5)
B = randn(5, 5, 5)
C = randn(5, 5, 5)
D = zeros(5, 5, 5)

@tensor backend = tblis begin
    D2[a, b, c] = A[a, e, f, c, f, g] * B[g, b, e] + α * C[c, a, b]
    E2[a, b, c] := A[a, e, f, c, f, g] * B[g, b, e] + α * C[c, a, b]
end
```

Additionally, the number of threads used by tblis can be set by:
```julia
using TensorOperationsTBLIS
tblis_set_num_threads(4)
@show tblis_get_num_threads()
```
