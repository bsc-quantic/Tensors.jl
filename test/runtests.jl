using Test
using Tensors
using OMEinsum

@testset "Unit tests" verbose = true begin
    include("Tensor_test.jl")
    include("Metadata_test.jl")
    include("Numerics_test.jl")
end

@testset "Integration tests" verbose = true begin
    include("integration/ChainRulesCore_test.jl")
    include("integration/BlockArray_test.jl")
end

if haskey(ENV, "ENABLE_AQUA_TESTS")
    @testset "Aqua" verbose = true begin
        using Aqua
        Aqua.test_all(Tensors, ambiguities = false)
    end
end
