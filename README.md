# TensorOperationsTBLIS.jl

Julia wrapper for [TBLIS](https://github.com/devinamatthews/tblis) with [TensorOperations.jl](https://github.com/Jutho/TensorOperations.jl).

[![CI][ci-img]][ci-url] [![CI (Julia nightly)][ci-julia-nightly-img]][ci-julia-nightly-url] [![][codecov-img]][codecov-url]

[ci-img]: https://github.com/lkdvos/TensorOperationsTBLIS.jl/actions/workflows/ci.yml/badge.svg
[ci-url]: https://github.com/lkdvos/TensorOperationsTBLIS.jl/actions/workflows/ci.yml

[ci-julia-nightly-img]: https://github.com/lkdvos/TensorOperationsTBLIS.jl/actions/workflows/ci-julia-nightly.yml/badge.svg
[ci-julia-nightly-url]: https://github.com/lkdvos/TensorOperationsTBLIS.jl/actions/workflows/ci-julia-nightly.yml

[codecov-img]: https://codecov.io/gh/lkdvos/TensorOperationsTBLIS.jl/graph/badge.svg?token=R86L0S70VT
[codecov-url]: https://codecov.io/gh/lkdvos/TensorOperationsTBLIS.jl

Currently provides implementations of `tensorcontract!`, `tensoradd!` and `tensortrace!` for array types compatible with Strided.jl, i.e. `StridedView{<:BlasFloat}`.
These can be accessed through the backend system of TensorOperations, i.e.

```julia
using TensorOperations
using TensorOperationsTBLIS

tblisbackend = TBLIS()
α = randn()
A = randn(5, 5, 5, 5, 5, 5)
B = randn(5, 5, 5)
C = randn(5, 5, 5)
D = zeros(5, 5, 5)

@tensor backend = tblisbackend begin
    D2[a, b, c] = A[a, e, f, c, f, g] * B[g, b, e] + α * C[c, a, b]
    E2[a, b, c] := A[a, e, f, c, f, g] * B[g, b, e] + α * C[c, a, b]
end
```

Additionally, the number of threads used by TBLIS can be set by:

```julia
TensorOperationsTBLIS.set_num_threads(4)
@show TensorOperationsTBLIS.get_num_threads()
```

## Notes

- This implementation of TBLIS for TensorOperations.jl is only supported from v5 of
  TensorOperations.jl onwards. For v4, an earlier version of this package exists.
  For older versions, you could look for
  [BliContractor.jl](https://github.com/xrq-phys/BliContractor.jl) or
  [TBLIS.jl](https://github.com/FermiQC/TBLIS.jl).
