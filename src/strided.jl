#-------------------------------------------------------------------------------------------
# Force strided implementation on AbstractArray instances with TBLIS backend
#-------------------------------------------------------------------------------------------
const SV = StridedView
function TensorOperations.tensoradd!(C::AbstractArray,
                                     A::AbstractArray, pA::Index2Tuple, conjA::Bool,
                                     α::Number, β::Number,
                                     backend::TBLIS, allocator=DefaultAllocator())
    # resolve conj flags and absorb into StridedView constructor to avoid type instabilities later on
    if conjA
        stridedtensoradd!(SV(C), conj(SV(A)), pA, α, β, backend, allocator)
    else
        stridedtensoradd!(SV(C), SV(A), pA, α, β, backend, allocator)
    end
    return C
end

function TensorOperations.tensortrace!(C::AbstractArray,
                                       A::AbstractArray, p::Index2Tuple, q::Index2Tuple,
                                       conjA::Bool,
                                       α::Number, β::Number,
                                       backend::TBLIS, allocator=DefaultAllocator())
    # resolve conj flags and absorb into StridedView constructor to avoid type instabilities later on
    if conjA
        stridedtensortrace!(SV(C), conj(SV(A)), p, q, α, β, backend, allocator)
    else
        stridedtensortrace!(SV(C), SV(A), p, q, α, β, backend, allocator)
    end
    return C
end

function TensorOperations.tensorcontract!(C::AbstractArray,
                                          A::AbstractArray, pA::Index2Tuple, conjA::Bool,
                                          B::AbstractArray, pB::Index2Tuple, conjB::Bool,
                                          pAB::Index2Tuple,
                                          α::Number, β::Number,
                                          backend::TBLIS, allocator=DefaultAllocator())
    # resolve conj flags and absorb into StridedView constructor to avoid type instabilities later on
    if conjA && conjB
        stridedtensorcontract!(SV(C), conj(SV(A)), pA, conj(SV(B)), pB, pAB, α, β,
                               backend, allocator)
    elseif conjA
        stridedtensorcontract!(SV(C), conj(SV(A)), pA, SV(B), pB, pAB, α, β,
                               backend, allocator)
    elseif conjB
        stridedtensorcontract!(SV(C), SV(A), pA, conj(SV(B)), pB, pAB, α, β,
                               backend, allocator)
    else
        stridedtensorcontract!(SV(C), SV(A), pA, SV(B), pB, pAB, α, β,
                               backend, allocator)
    end
    return C
end

#-------------------------------------------------------------------------------------------
# StridedView implementation
#-------------------------------------------------------------------------------------------
function tblis_tensor(A::StridedView,
                      s::Number=one(eltype(A)),
                      szA::Vector{len_type}=collect(len_type, size(A)),
                      strA::Vector{stride_type}=collect(stride_type, strides(A)))
    t₁ = LibTBLIS.tblis_tensor(A, s, szA, strA)
    if A.op == conj
        t₂ = LibTBLIS.tblis_tensor(t₁.type, Cint(1), t₁.scalar, t₁.data, t₁.ndim, t₁.len,
                                   t₁.stride)
        return t₂
    else
        return t₁
    end
end

function TensorOperations.stridedtensoradd!(C::StridedView{T},
                                            A::StridedView{T}, pA::Index2Tuple,
                                            α::Number, β::Number,
                                            backend::TBLIS,
                                            allocator=DefaultAllocator()) where {T<:BlasFloat}
    argcheck_tensoradd(C, A, pA)
    dimcheck_tensoradd(C, A, pA)
    if Base.mightalias(C, A)
        throw(ArgumentError("output tensor must not be aliased with input tensor"))
    end

    C_tblis = tblis_tensor(C, β)
    A_tblis = tblis_tensor(A, α)
    einA, einC = TensorOperations.add_labels(pA)
    tblis_tensor_add(A_tblis, string(einA...), C_tblis, string(einC...))
    return C
end

function TensorOperations.stridedtensortrace!(C::StridedView{T},
                                              A::StridedView{T},
                                              p::Index2Tuple,
                                              q::Index2Tuple,
                                              α::Number, β::Number,
                                              backend::TBLIS,
                                              allocator=DefaultAllocator()) where {T<:BlasFloat}
    argcheck_tensortrace(C, A, p, q)
    dimcheck_tensortrace(C, A, p, q)

    Base.mightalias(C, A) &&
        throw(ArgumentError("output tensor must not be aliased with input tensor"))

    C_tblis = tblis_tensor(C, β)
    A_tblis = tblis_tensor(A, α)
    einA, einC = TensorOperations.trace_labels(p, q)
    tblis_tensor_add(A_tblis, string(einA...), C_tblis, string(einC...))
    return C
end

function TensorOperations.stridedtensorcontract!(C::StridedView{T},
                                                 A::StridedView{T}, pA::Index2Tuple,
                                                 B::StridedView{T}, pB::Index2Tuple,
                                                 pAB::Index2Tuple,
                                                 α::Number, β::Number,
                                                 backend::TBLIS,
                                                 allocator=DefaultAllocator()) where {T<:BlasFloat}
    argcheck_tensorcontract(C, A, pA, B, pB, pAB)
    dimcheck_tensorcontract(C, A, pA, B, pB, pAB)

    (Base.mightalias(C, A) || Base.mightalias(C, B)) &&
        throw(ArgumentError("output tensor must not be aliased with input tensor"))

    einA, einB, einC = TensorOperations.contract_labels(pA, pB, pAB)

    # tblis_tensor_mult ignores conjugation flags in A and B (and C)
    if A.op == conj && B.op == conj
        iszero(β) || conj!(C)

        C_tblis = tblis_tensor(C, conj(β))
        A_tblis = tblis_tensor(A, conj(α))
        B_tblis = tblis_tensor(B)

        tblis_tensor_mult(A_tblis, string(einA...), B_tblis, string(einB...),
                          C_tblis, string(einC...))
        conj!(C)
    elseif A.op == conj
        pA2 = TensorOperations.trivialpermutation(pA)
        A2 = StridedView(TensorOperations.tensoralloc_add(eltype(A), A, pA2, false,
                                                          Val(true),
                                                          allocator))
        A2 = tensorcopy!(A2, A, pA2, false, α, backend, allocator)

        C_tblis = tblis_tensor(C, β)
        A_tblis = tblis_tensor(A2)
        B_tblis = tblis_tensor(B)

        tblis_tensor_mult(A_tblis, string(einA...), B_tblis, string(einB...),
                          C_tblis, string(einC...))
    elseif B.op == conj
        pB2 = TensorOperations.trivialpermutation(pB)
        B2 = StridedView(TensorOperations.tensoralloc_add(eltype(B), B, pB2, false,
                                                          Val(true),
                                                          allocator))
        B2 = tensorcopy!(B2, B, pB2, false, α, backend, allocator)

        C_tblis = tblis_tensor(C, β)
        A_tblis = tblis_tensor(A)
        B_tblis = tblis_tensor(B2)

        tblis_tensor_mult(A_tblis, string(einA...), B_tblis, string(einB...),
                          C_tblis, string(einC...))
    else
        C_tblis = tblis_tensor(C, β)
        A_tblis = tblis_tensor(A, α)
        B_tblis = tblis_tensor(B)

        tblis_tensor_mult(A_tblis, string(einA...), B_tblis, string(einB...),
                          C_tblis, string(einC...))
    end
    return C
end
