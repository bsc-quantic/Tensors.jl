@testset "Dagger" begin
    using Tensors: Tensor, contract, permutedims, svd
    using Dagger: spawn, chunks, Distribute, Blocks, distribute, compute, collect, DArray

    @testset "distribute" begin
        @testset "Tensor" begin
            data = rand(4, 4)
            chunk_sizes = (1, 2)
            darray = distribute(data, Blocks(chunk_sizes...))
            indices = (:i, :j)

            # Create a dagger_tensor
            tensor = Tensor(data, indices)
            dagger_tensor = Tensor(darray, indices)

            @test labels(dagger_tensor) == labels(tensor)
            @test parent(dagger_tensor) |> compute |> collect ≈ parent(tensor)
            @test parent(dagger_tensor) |> chunks |> size == (4, 2)
        end

        # @testset "permutedims" begin
        #     data = rand(4, 4, 4)
        #     chunk_sizes = (2, 1, 2)
        #     darray = distribute(data, Blocks(chunk_sizes...))
        #     indices = (:i, :j, :k)
        #     perm = (3, 1, 2)

        #     tensor = Tensor(data, indices)
        #     dagger_tensor = Tensor(darray, indices)

        #     permuted_tensor = permutedims(tensor, perm)
        #     permuted_dagger_tensor = permutedims(dagger_tensor, perm)

        #     @test parent(permuted_dagger_tensor) isa DArray
        #     @test parent(permuted_dagger_tensor) |> chunks |> size == (2, 2, 2)
        #     @test labels(permuted_dagger_tensor) == labels(permuted_tensor)
        #     @test parent(permuted_dagger_tensor) |> compute |> collect ≈ parent(permuted_tensor)
        # end

        @testset "contract" begin
            @testset "block-block" begin
                data1, data2 = rand(4, 4), rand(4, 4)
                chunk_sizes1, chunk_sizes2 = (1, 2), (2, 2)
                darray1 = distribute(data1, Blocks(chunk_sizes1...))
                darray2 = distribute(data2, Blocks(chunk_sizes2...))

                tensor1 = Tensor(data1, [:i, :j])
                tensor2 = Tensor(data2, [:j, :k])
                dagger_tensor1 = Tensor(darray1, [:i, :j])
                dagger_tensor2 = Tensor(darray2, [:j, :k])

                contracted_tensor = contract(tensor1, tensor2)
                contracted_dagger_tensor = contract(dagger_tensor1, dagger_tensor2)

                @test parent(contracted_dagger_tensor) |> compute isa DArray
                @test contracted_dagger_tensor |> labels == (:i, :k)
                @test parent(contracted_dagger_tensor) |> compute |> chunks |> size == (4, 2)
                @test parent(contracted_dagger_tensor) |> compute |> collect ≈ parent(contracted_tensor)
            end

            # @testset "block-unblock" begin
            #     data1, data2 = rand(4, 4), rand(4, 4)
            #     chunk_sizes = (1, 2)
            #     darray = distribute(data2, Blocks(chunk_sizes...))

            #     tensor = Tensor(data1, [:i, :j])
            #     dagger_tensor = Tensor(darray, [:j, :k])

            #     contracted_tensor = contract(tensor, dagger_tensor)

            #     @test contracted_tensor |> labels == (:i, :k)
            #     @test (contracted_tensor |> parent |> chunks)[2] == [2, 2]
            #     @test parent(contracted_tensor) |> compute |> collect ≈ parent(contract(tensor, Tensor(data2, [:j, :k])))
            # end
        end

        # It seems that svd, eigen and qr are not yet supported for Dagger.DArray:

        # @testset "svd" begin
        #     data = rand(4, 4, 4)
        #     chunk_sizes = (2, 1, 2)
        #     darray = distribute(data, Blocks(chunk_sizes...))
        #     indices = (:i, :j, :k)

        #     tensor = Tensor(data, indices)
        #     dagger_tensor = Tensor(darray, indices)

        #     U, S, V = svd(tensor; left_inds = (:i, :j))
        #     U̅, S̅, V̅ = svd(dagger_tensor; left_inds = (:i, :j))

        #     @test parent(U̅) |> compute |> collect ≈ parent(U)
        #     @test parent(S̅) |> compute |> collect ≈ parent(S)
        #     @test parent(V̅) |> compute |> collect ≈ parent(V)
        # end

        # using LinearAlgebra: eigen, qr

        # TODO: Using LinearAlgebra since `eigen` is not yet supported in `Tensors`
        # @testset "eigendecomposition" begin
        #     data = rand(4, 4)
        #     data = (data + data') / 2  # Make the matrix symmetric
        #     chunk_sizes = (2, 2)
        #     darray = distribute(data, Blocks(chunk_sizes...))

        #     eigen_decomp = eigen(data)
        #     eigen_block_decomp = eigen(darray)

        #     @test collect(compute(eigen_block_decomp)).vectors ≈ eigen_decomp.vectors
        #     @test collect(compute(eigen_block_decomp)).values ≈ eigen_decomp.values
        # end

        # TODO: Using LinearAlgebra since `qr` is not yet supported in `Tensors`
        # @testset "QR decomposition" begin
        #     data = rand(4, 4)
        #     chunk_sizes = (2, 2)
        #     darray = distribute(data, Blocks(chunk_sizes...))

        #     qr_decomp = qr(data)
        #     qr_block_decomp = qr(darray)

        #     @test collect(compute(qr_block_decomp)).Q ≈ qr_decomp.Q
        #     @test collect(compute(qr_block_decomp)).R ≈ qr_decomp.R
        # end
    end

    @testset "delayed" begin
        @testset "Tensor" begin
            data = rand(4, 4)
            indices = (:i, :j)

            # Create a delayed_tensor
            tensor = Tensor(data, indices)
            delayed_tensor = delayed(Tensor)(data, indices)

            @test delayed_tensor |> compute |> collect |> labels == labels(tensor)
            @test delayed_tensor |> compute |> collect |> parent ≈ parent(tensor)
        end

        @testset "permutedims" begin
            data = rand(4, 4, 4)
            indices = (:i, :j, :k)
            perm = (3, 1, 2)

            tensor = Tensor(data, indices)
            delayed_tensor = delayed(Tensor)(data, indices)

            permuted_tensor = permutedims(tensor, perm)
            permuted_delayed_tensor = delayed(permutedims)(delayed_tensor, perm)

            @test permuted_delayed_tensor |> compute |> collect |> labels == labels(permuted_tensor)
            @test permuted_delayed_tensor |> compute |> collect |> parent ≈ parent(permuted_tensor)
        end

        @testset "contract" begin
            @testset "block-block" begin
                data1, data2 = rand(4, 4), rand(4, 4)
                chunk_sizes1, chunk_sizes2 = (1, 2), (2, 2)
                darray1 = distribute(data1, Blocks(chunk_sizes1...))
                darray2 = distribute(data2, Blocks(chunk_sizes2...))

                tensor1 = Tensor(data1, [:i, :j])
                tensor2 = Tensor(data2, [:j, :k])
                dagger_tensor1 = Tensor(darray1, [:i, :j])
                dagger_tensor2 = Tensor(darray2, [:j, :k])

                contracted_tensor = contract(tensor1, tensor2)
                contracted_dagger_tensor = contract(dagger_tensor1, dagger_tensor2)

                @test parent(contracted_dagger_tensor) |> compute isa DArray
                @test contracted_dagger_tensor |> labels == (:i, :k)
                @test parent(contracted_dagger_tensor) |> compute |> chunks |> size == (4, 2)
                @test parent(contracted_dagger_tensor) |> compute |> collect ≈ parent(contracted_tensor)
            end
        end
    end
end