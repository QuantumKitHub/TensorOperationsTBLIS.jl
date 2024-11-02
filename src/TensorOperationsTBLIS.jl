module TensorOperationsTBLIS

using TensorOperations
using TensorOperations: StridedView, DefaultAllocator, IndexError
using TensorOperations: istrivialpermutation, BlasFloat, linearize
using TensorOperations: argcheck_tensoradd, dimcheck_tensoradd,
                        argcheck_tensortrace, dimcheck_tensortrace,
                        argcheck_tensorcontract, dimcheck_tensorcontract
using TensorOperations: Index2Tuple, IndexTuple, linearize, IndexError
using LinearAlgebra: BlasFloat
using TupleTools

include("LibTBLIS.jl")
using .LibTBLIS
using .LibTBLIS: LibTBLIS, len_type, stride_type

export TBLIS
export get_num_tblis_threads, set_num_tblis_threads

get_num_tblis_threads() = convert(Int, LibTBLIS.tblis_get_num_threads())
set_num_tblis_threads(n) = LibTBLIS.tblis_set_num_threads(convert(Cuint, n))

# TensorOperations
#------------------

struct TBLIS <: TensorOperations.AbstractBackend end

Base.@deprecate(tblisBackend(), TBLIS())
Base.@deprecate(tblis_get_num_threads(), get_num_tblis_threads())
Base.@deprecate(tblis_set_num_threads(n), set_num_tblis_threads(n))

include("strided.jl")

end # module TensorOperationsTBLIS
