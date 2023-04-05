using Test
using Tensors

@testset "Unit tests" verbose = true begin
    include("Tensor_test.jl")
    include("Metadata_test.jl")
    include("Differentiation_test.jl")
    include("Numerics_test.jl")
end

if haskey(ENV, "ENABLE_AQUA_TESTS")
    @testset "Aqua" verbose = true begin
        using Aqua
        Aqua.test_all(Tensors, ambiguities=false)
    end
end
