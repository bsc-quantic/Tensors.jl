@testset "Numerics" begin
    using Tensors: Tensor

    @testset "svd" begin
        using LinearAlgebra

        data = rand(2, 2, 2)
        tensor = Tensor(data, (:i, :j, :k))
        # Throw exception if left_inds is not provided
        @test_throws UndefVarError svd(tensor)
        # Throw expcetion if left_inds ∉ labels(tensor)
        @test_throws ErrorException svd(tensor, (:l,))

        U, s, V = svd(tensor, labels(tensor)[1:2])
        @test labels(U)[1:2] == labels(tensor)[1:2]
        @test labels(U)[3] == labels(s)[1]
        @test labels(V)[1] == labels(s)[2]
        @test labels(V)[2] == labels(tensor)[3]

        @test isapprox(U * s * V, data)
    end
end