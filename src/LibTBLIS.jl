module LibTBLIS

using tblis_jll
using LinearAlgebra: BlasFloat

export tblis_scalar, tblis_tensor
export tblis_tensor_add, tblis_tensor_mult, tblis_tensor_dot
export tblis_set_num_threads, tblis_get_num_threads

const ptrdiff_t = Cptrdiff_t

const scomplex = ComplexF32
const dcomplex = ComplexF64

const IS_LIBC_MUSL = occursin("musl", Base.BUILD_TRIPLET)
if Sys.isapple() && Sys.ARCH === :aarch64
    include("lib/aarch64-apple-darwin20.jl")
elseif Sys.islinux() && Sys.ARCH === :aarch64 && !IS_LIBC_MUSL
    include("lib/aarch64-linux-gnu.jl")
elseif Sys.islinux() && Sys.ARCH === :aarch64 && IS_LIBC_MUSL
    include("lib/aarch64-linux-musl.jl")
elseif Sys.islinux() && startswith(string(Sys.ARCH), "arm") && !IS_LIBC_MUSL
    include("lib/armv7l-linux-gnueabihf.jl")
elseif Sys.islinux() && startswith(string(Sys.ARCH), "arm") && IS_LIBC_MUSL
    include("lib/armv7l-linux-musleabihf.jl")
elseif Sys.islinux() && Sys.ARCH === :i686 && !IS_LIBC_MUSL
    include("lib/i686-linux-gnu.jl")
elseif Sys.islinux() && Sys.ARCH === :i686 && IS_LIBC_MUSL
    include("lib/i686-linux-musl.jl")
elseif Sys.iswindows() && Sys.ARCH === :i686
    include("lib/i686-w64-mingw32.jl")
elseif Sys.islinux() && Sys.ARCH === :powerpc64le
    include("lib/powerpc64le-linux-gnu.jl")
elseif Sys.isapple() && Sys.ARCH === :x86_64
    include("lib/x86_64-apple-darwin14.jl")
elseif Sys.islinux() && Sys.ARCH === :x86_64 && !IS_LIBC_MUSL
    include("lib/x86_64-linux-gnu.jl")
elseif Sys.islinux() && Sys.ARCH === :x86_64 && IS_LIBC_MUSL
    include("lib/x86_64-linux-musl.jl")
elseif Sys.isbsd() && !Sys.isapple()
    include("lib/x86_64-unknown-freebsd.jl")
elseif Sys.iswindows() && Sys.ARCH === :x86_64
    include("lib/x86_64-w64-mingw32.jl")
else
    error("Unknown platform: $(Base.BUILD_TRIPLET)")
end

# tblis_scalar and tblis_tensor
# -----------------------------
"""
    tblis_scalar(s::Number)

Initializes a tblis scalar from a number.
"""
function tblis_scalar end

"""
    tblis_tensor(A::AbstractArray{T<:BlasFloat}, [szA::Vector{Int}, strA::Vector{Int}, scalar::Number])

Initializes a tblis tensor from an array that should be strided and admit a pointer to its
data. This operation is deemed unsafe, in the sense that the user is responsible for ensuring
that the reference to the array and the sizes and strides stays alive during the lifetime of
the tensor.
"""
function tblis_tensor end

for (T, tblis_init_scalar, tblis_init_tensor, tblis_init_tensor_scaled) in
    ((:Float32, :tblis_init_scalar_s, :tblis_init_tensor_s, :tblis_init_tensor_scaled_s),
     (:Float64, :tblis_init_scalar_d, :tblis_init_tensor_d, :tblis_init_tensor_scaled_d),
     (:ComplexF32, :tblis_init_scalar_c, :tblis_init_tensor_c, :tblis_init_tensor_scaled_c),
     (:ComplexF64, :tblis_init_scalar_z, :tblis_init_tensor_z, :tblis_init_tensor_scaled_z))
    @eval begin
        function tblis_scalar(s::$T)
            t = Ref{tblis_scalar}()
            $tblis_init_scalar(t, s)
            return t[]
        end
        function tblis_tensor(A::AbstractArray{$T,N},
                              s::Number=one(T),
                              szA::Vector{len_type}=collect(len_type, size(A)),
                              strA::Vector{stride_type}=collect(stride_type, strides(A))) where {N}
            t = Ref{tblis_tensor}()
            if isone(s)
                $tblis_init_tensor(t, N, pointer(szA), pointer(A), pointer(strA))
            else
                $tblis_init_tensor_scaled(t, $T(s), N, pointer(szA), pointer(A),
                                          pointer(strA))
            end
            return t[]
        end
    end
end

# tensor operations
# ------------------
function tblis_tensor_add(A::tblis_tensor, idxA, B::tblis_tensor, idxB)
    return tblis_tensor_add(C_NULL, C_NULL, Ref(A), idxA, Ref(B), idxB)
end

function tblis_tensor_mult(A::tblis_tensor, idxA, B::tblis_tensor, idxB, C::tblis_tensor,
                           idxC)
    return tblis_tensor_mult(C_NULL, C_NULL, Ref(A), idxA, Ref(B), idxB, Ref(C), idxC)
end

function tblis_tensor_dot(A::tblis_tensor, idxA, B::tblis_tensor, idxB, C::tblis_scalar)
    return tblis_tensor_dot(C_NULL, C_NULL, Ref(A), idxA, Ref(B), idxB, Ref(C))
end

end
