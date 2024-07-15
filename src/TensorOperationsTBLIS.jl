module TensorOperationsTBLIS

using TensorOperations
using TensorOperations: Index2Tuple, IndexTuple, linearize, IndexError
using LinearAlgebra: BlasFloat, rmul!
using TupleTools

include("LibTblis.jl")
using .LibTblis

export tblis_set_num_threads, tblis_get_num_threads
export tblisBackend

# TensorOperations
#------------------

struct tblisBackend <: TensorOperations.AbstractBackend end

function TensorOperations.tensoradd!(C::StridedArray{T}, A::StridedArray{T},
                                     pA::Index2Tuple, conjA::Bool,
                                     α::Number, β::Number,
                                     ::tblisBackend) where {T<:BlasFloat}
    TensorOperations.argcheck_tensoradd(C, A, pA)
    TensorOperations.dimcheck_tensoradd(C, A, pA)

    szC = collect(size(C))
    strC = collect(strides(C))
    C_tblis = tblis_tensor(C, szC, strC, β)

    szA = collect(size(A))
    strA = collect(strides(A))
    A_tblis = tblis_tensor(conjA ? conj(A) : A, szA, strA, α)

    einA, einC = TensorOperations.add_labels(pA)
    tblis_tensor_add(A_tblis, string(einA...), C_tblis, string(einC...))

    return C
end

function TensorOperations.tensorcontract!(C::StridedArray{T},
                                          A::StridedArray{T}, pA::Index2Tuple,
                                          conjA::Bool, B::StridedArray{T},
                                          pB::Index2Tuple, conjB::Bool, pAB::Index2Tuple,
                                          α::Number, β::Number,
                                          ::tblisBackend) where {T<:BlasFloat}
    TensorOperations.argcheck_tensorcontract(C, A, pA, B, pB, pAB)
    TensorOperations.dimcheck_tensorcontract(C, A, pA, B, pB, pAB)

    rmul!(C, β) # TODO: is it possible to use tblis scaling here?
    szC = ndims(C) == 0 ? Int[] : collect(size(C))
    strC = ndims(C) == 0 ? Int[] : collect(strides(C))
    C_tblis = tblis_tensor(C, szC, strC)

    szA = collect(size(A))
    strA = collect(strides(A))
    A_tblis = tblis_tensor(conjA ? conj(A) : A, szA, strA, α)

    szB = collect(size(B))
    strB = collect(strides(B))
    B_tblis = tblis_tensor(conjB ? conj(B) : B, szB, strB, 1)

    einA, einB, einC = TensorOperations.contract_labels(pA, pB, pAB)
    tblis_tensor_mult(A_tblis, string(einA...), B_tblis, string(einB...), C_tblis,
                      string(einC...))

    return C
end

function TensorOperations.tensortrace!(C::StridedArray{T},
                                       A::StridedArray{T}, p::Index2Tuple, q::Index2Tuple,
                                       conjA::Bool,
                                       α::Number, β::Number,
                                       ::tblisBackend) where {T<:BlasFloat}
    TensorOperations.argcheck_tensortrace(C, A, p, q)
    TensorOperations.dimcheck_tensortrace(C, A, p, q)

    rmul!(C, β) # TODO: is it possible to use tblis scaling here?
    szC = ndims(C) == 0 ? Int[] : collect(size(C))
    strC = ndims(C) == 0 ? Int[] : collect(strides(C))
    C_tblis = tblis_tensor(C, szC, strC)

    szA = collect(size(A))
    strA = collect(strides(A))
    A_tblis = tblis_tensor(conjA ? conj(A) : A, szA, strA, α)

    einA, einC = TensorOperations.trace_labels(p, q)

    tblis_tensor_add(A_tblis, string(einA...), C_tblis, string(einC...))

    return C
end

end # module TensorOperationsTBLIS
