using TensorOperations, TensorOperationsTBLIS
using TensorOperations: IndexError
using TensorOperations: DefaultAllocator, ManualAllocator
using Test, Random, LinearAlgebra
Random.seed!(1234567)

@testset "TensorOperationsTBLIS.jl" begin
    TensorOperationsTBLIS.set_num_threads(1)
    @testset "method syntax" verbose = true begin
        include("methods.jl")
    end

    @test TensorOperationsTBLIS.get_num_threads() == 1
    TensorOperationsTBLIS.set_num_threads(2)
    @testset "macro with index notation" verbose = true begin
        include("tensor.jl")
    end

    @test TensorOperationsTBLIS.get_num_threads() == 2
end
