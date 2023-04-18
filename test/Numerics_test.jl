@testset "Numerics" begin
    using Tensors: Tensor

    @testset "svd" begin
        using LinearAlgebra

        data = rand(2, 2, 2)
        tensor = Tensor(data, (:i, :j, :k))

        @testset "Error Handling Test" begin
            # Throw exception if left_inds is not provided
            @test_throws ErrorException svd(tensor)
            # Throw exception if left_inds ∉ labels(tensor)
            @test_throws ErrorException svd(tensor, (:l,))
        end

        @testset "Labels Test" begin
            U, s, V = svd(tensor, labels(tensor)[1:2])
            @test labels(U)[1:2] == labels(tensor)[1:2]
            @test labels(U)[3] == labels(s)[1]
            @test labels(V)[1] == labels(s)[2]
            @test labels(V)[2] == labels(tensor)[3]
        end

        @testset "Size Test" begin
            U, s, V = svd(tensor, labels(tensor)[1:2])
            @test size(U) == (2, 2, 2)
            @test size(s) == (2, 2)
            @test size(V) == (2, 2)

            # Additional test with different dimensions
            data2 = rand(2, 4, 6, 8)
            tensor2 = Tensor(data2, (:i, :j, :k, :l))
            U2, s2, V2 = svd(tensor2, labels(tensor2)[1:2])
            @test size(U2) == (2, 4, 8)
            @test size(s2) == (8, 8)
            @test size(V2) == (8, 6, 8)
        end

        @testset "Accuracy Test" begin
            U, s, V = svd(tensor, labels(tensor)[1:2])
            @test U * s * V ≈ tensor

            data2 = rand(2, 4, 6, 8)
            tensor2 = Tensor(data2, (:i, :j, :k, :l))
            U2, s2, V2 = svd(tensor2, labels(tensor2)[1:2])
            @test U2 * s2 * V2 ≈ tensor2
        end
    end
end
