using Test
using Tensors

@testset "Unit tests" verbose = true begin
    include("Tensor_test.jl")
    include("Metadata_test.jl")
    include("Differentiation_test.jl")
end