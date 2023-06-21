@testset "Numerics" begin
    using LinearAlgebra

    @testset "svd" begin
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

    @testset "contract" begin
        using Tensors: Tensor, contract

        @testset "axis sum" begin
            A = Tensor(rand(2, 3, 4), (:i, :j, :k))

            C = contract(A, dims = (:i,))
            C_ein = ein"ijk -> jk"(A)
            @test labels(C) == (:j, :k)
            @test size(C) == size(C_ein) == (3, 4)
            @test C ≈ C_ein
        end

        @testset "diagonal" begin
            A = Tensor(rand(2, 3, 2), (:i, :j, :i))

            C = contract(A, dims = ())
            C_ein = ein"iji -> ij"(A)
            @test labels(C) == (:i, :j)
            @test size(C) == size(C_ein) == (2, 3)
            @test C ≈ C_ein
        end

        @testset "trace" begin
            A = Tensor(rand(2, 3, 2), (:i, :j, :i))

            C = contract(A, dims = (:i,))
            C_ein = ein"iji -> j"(A)
            @test labels(C) == (:j,)
            @test size(C) == size(C_ein) == (3,)
            @test C ≈ C_ein
        end

        @testset "matrix multiplication" begin
            A = Tensor(rand(2, 3), (:i, :j))
            B = Tensor(rand(3, 4), (:j, :k))

            C = contract(A, B)
            C_mat = parent(A) * parent(B)
            @test labels(C) == (:i, :k)
            @test size(C) == (2, 4) == size(C_mat)
            @test C ≈ A * B ≈ C_mat
        end

        @testset "inner product" begin
            A = Tensor(rand(3, 4), (:i, :j))
            B = Tensor(rand(4, 3), (:j, :i))

            C = contract(A, B)
            C_res = LinearAlgebra.tr(parent(A) * parent(B))
            @test labels(C) == ()
            @test size(C) == () == size(C_res)
            @test only(C) ≈ C_res
        end

        @testset "outer product" begin
            A = Tensor(rand(2, 2), (:i, :j))
            B = Tensor(rand(2, 2), (:k, :l))

            C = contract(A, B)
            C_ein = ein"ij, kl -> ijkl"(A, B)
            @test size(C) == (2, 2, 2, 2) == size(C_ein)
            @test labels(C) == (:i, :j, :k, :l)
            @test C ≈ C_ein
        end

        @testset "scale" begin
            A = Tensor(rand(2, 2), (:i, :j))
            scalar = 2.0

            C = contract(A, scalar)
            @test labels(C) == (:i, :j)
            @test size(C) == (2, 2)
            @test C ≈ parent(A) * scalar

            D = contract(scalar, A)
            @test labels(D) == (:i, :j)
            @test size(D) == (2, 2)
            @test D ≈ scalar * parent(A)
        end

        @testset "manual" begin
            A = Tensor(rand(2, 3, 4), (:i, :j, :k))
            B = Tensor(rand(4, 5, 3), (:k, :l, :j))

            # Contraction of all common indices
            C = contract(A, B, dims = (:j, :k))
            C_ein = ein"ijk, klj -> il"(A, B)
            @test labels(C) == (:i, :l)
            @test size(C) == (2, 5) == size(C_ein)
            @test C ≈ C_ein

            # Contraction of not all common indices
            C = contract(A, B, dims = (:j,))
            C_ein = ein"ijk, klj -> ikl"(A, B)
            @test labels(C) == (:i, :k, :l)
            @test size(C) == (2, 4, 5) == size(C_ein)
            @test C ≈ C_ein

            @testset "Complex numbers" begin
                A = Tensor(rand(Complex{Float64}, 2, 3, 4), (:i, :j, :k))
                B = Tensor(rand(Complex{Float64}, 4, 5, 3), (:k, :l, :j))

                C = contract(A, B, dims = (:j, :k))
                C_ein = ein"ijk, klj -> il"(A, B)
                @test labels(C) == (:i, :l)
                @test size(C) == (2, 5) == size(C_ein)
                @test C ≈ C_ein
            end
        end
    end

    @testset "qr" begin
        data = rand(2, 2, 2)
        tensor = Tensor(data, (:i, :j, :k))

        @testset "[exceptions]" begin
            # Throw exception if left_inds is not provided
            @test_throws ErrorException qr(tensor)
            # Throw exception if left_inds ∉ labels(tensor)
            @test_throws ErrorException qr(tensor, (:l,))
            # throw exception if no right-inds
            @test_throws ErrorException qr(tensor, (:i, :j, :k))
        end

        @testset "labels" begin
            Q, R = qr(tensor, labels(tensor)[1:2])
            @test labels(Q)[1:2] == labels(tensor)[1:2]
            @test labels(Q)[3] == labels(R)[1]
            @test labels(R)[2] == labels(tensor)[3]
        end

        @testset "size" begin
            Q, R = qr(tensor, labels(tensor)[1:2])
            # Q's new index size = min(prod(left_inds), prod(right_inds)).
            @test size(Q) == (2, 2, 4)
            @test size(R) == (2, 2)

            # Additional test with different dimensions
            data2 = rand(2, 4, 6, 8)
            tensor2 = Tensor(data2, (:i, :j, :k, :l))
            Q2, R2 = qr(tensor2, labels(tensor2)[1:2])
            @test size(Q2) == (2, 4, 8)
            @test size(R2) == (8, 6, 8)
        end

        @testset "[accuracy]" begin
            Q, R = qr(tensor, labels(tensor)[1:2])
            Q_truncated = view(Q, labels(Q)[end] => 1:2)
            tensor_recovered = ein"ijk, kl -> ijl"(Q_truncated, R)
            @test tensor_recovered ≈ tensor

            data2 = rand(2, 4, 6, 8)
            tensor2 = Tensor(data2, (:i, :j, :k, :l))
            Q2, R2 = qr(tensor2, labels(tensor2)[1:2])
            tensor2_recovered = ein"ijk, klm -> ijlm"(Q2, R2)
            @test tensor2_recovered ≈ tensor2
        end
    end
end
