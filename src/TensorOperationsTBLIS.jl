module TensorOperationsTBLIS

using TensorOperations
using TensorOperations: Index2Tuple, IndexTuple, linearize, IndexError
using LinearAlgebra: BlasFloat, rmul!
using TupleTools

include("LibTblis.jl")
using .LibTblis

export tblis_set_num_threads, tblis_get_num_threads

# TensorOperations
#------------------

const tblisBackend = TensorOperations.Backend{:tblis}

function TensorOperations.tensoradd!(C::StridedArray{T}, pC::Index2Tuple,
                                     A::StridedArray{T}, conjA::Symbol,
                                     α::Number, β::Number,
                                     ::tblisBackend) where {T<:BlasFloat}
    TensorOperations.argcheck_tensoradd(C, pC, A)
    TensorOperations.dimcheck_tensoradd(C, pC, A)

    szC = collect(size(C))
    strC = collect(strides(C))
    C_tblis = tblis_tensor(C, szC, strC, β)

    szA = collect(size(A))
    strA = collect(strides(A))
    A_tblis = tblis_tensor(conjA == :C ? conj(A) : A, szA, strA, α)

    einA, einC = TensorOperations.add_labels(pC)
    tblis_tensor_add(A_tblis, string(einA...), C_tblis, string(einC...))

    return C
end

function TensorOperations.tensorcontract!(C::StridedArray{T}, pC::Index2Tuple,
                                          A::StridedArray{T}, pA::Index2Tuple,
                                          conjA::Symbol, B::StridedArray{T},
                                          pB::Index2Tuple, conjB::Symbol, α::Number,
                                          β::Number, ::tblisBackend) where {T<:BlasFloat}
    TensorOperations.argcheck_tensorcontract(C, pC, A, pA, B, pB)
    TensorOperations.dimcheck_tensorcontract(C, pC, A, pA, B, pB)

    rmul!(C, β) # TODO: is it possible to use tblis scaling here?
    szC = ndims(C) == 0 ? Int[] : collect(size(C))
    strC = ndims(C) == 0 ? Int[] : collect(strides(C))
    C_tblis = tblis_tensor(C, szC, strC)

    szA = collect(size(A))
    strA = collect(strides(A))
    A_tblis = tblis_tensor(conjA == :C ? conj(A) : A, szA, strA, α)

    szB = collect(size(B))
    strB = collect(strides(B))
    B_tblis = tblis_tensor(conjB == :C ? conj(B) : B, szB, strB, 1)

    einA, einB, einC = TensorOperations.contract_labels(pA, pB, pC)
    tblis_tensor_mult(A_tblis, string(einA...), B_tblis, string(einB...), C_tblis,
                      string(einC...))

    return C
end

function TensorOperations.tensortrace!(C::StridedArray{T}, pC::Index2Tuple,
                                       A::StridedArray{T}, pA::Index2Tuple, conjA::Symbol,
                                       α::Number, β::Number,
                                       ::tblisBackend) where {T<:BlasFloat}
    TensorOperations.argcheck_tensortrace(C, pC, A, pA)
    TensorOperations.dimcheck_tensortrace(C, pC, A, pA)

    rmul!(C, β) # TODO: is it possible to use tblis scaling here?
    szC = ndims(C) == 0 ? Int[] : collect(size(C))
    strC = ndims(C) == 0 ? Int[] : collect(strides(C))
    C_tblis = tblis_tensor(C, szC, strC)

    szA = collect(size(A))
    strA = collect(strides(A))
    A_tblis = tblis_tensor(conjA == :C ? conj(A) : A, szA, strA, α)

    einA, einC = TensorOperations.trace_labels(pC, pA...)

    tblis_tensor_add(A_tblis, string(einA...), C_tblis, string(einC...))

    return C
end

end # module TensorOperationsTBLIS
