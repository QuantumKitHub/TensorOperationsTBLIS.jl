using TensorOperations, TensorOperationsTBLIS
using TensorOperations: IndexError
using TensorOperations: DefaultAllocator, ManualAllocator
using Test, Random, LinearAlgebra
Random.seed!(1234567)

@testset "TensorOperationsTBLIS.jl" begin
    set_num_tblis_threads(1)
    @testset "method syntax" verbose = true begin
        include("methods.jl")
    end

    @test get_num_tblis_threads() == 1
    set_num_tblis_threads(2)
    @testset "macro with index notation" verbose = true begin
        include("tensor.jl")
    end

    @test get_num_tblis_threads() == 2
end
