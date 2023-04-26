@testset "BlockArray" begin
    using Tensors: Tensor, contract, permutedims
    using BlockArrays

    @testset "Tensor" begin
        data = rand(4, 4)
        block_sizes = ([2, 2], [2, 2])
        block_array = BlockArray(data, block_sizes...)
        indices = (:i, :j)

        # Create a block_tensor
        tensor = Tensor(data, indices)
        block_tensor = Tensor(block_array, indices)

        @test labels(block_tensor) == labels(tensor)
        @test Array(parent(block_tensor)) ≈ parent(tensor)
    end

    @testset "permutedims" begin
        data = rand(4, 4, 4)
        block_sizes = ([2, 2], [2, 2], [1, 3])
        block_array = BlockArray(data, block_sizes...)
        indices = (:i, :j, :k)
        perm = (3, 1, 2)

        tensor = Tensor(data, indices)
        block_tensor = Tensor(block_array, indices)

        permuted_tensor = permutedims(tensor, perm)
        permuted_block_tensor = permutedims(block_tensor, perm)

        @test parent(permuted_block_tensor) isa BlockArray
        @test parent(permuted_block_tensor) |> blocksizes == ([1, 3], [2, 2], [2, 2])
        @test labels(permuted_block_tensor) == labels(permuted_tensor)
        @test Array(parent(permuted_block_tensor)) ≈ parent(permuted_tensor)
    end

    @testset "contract block tensors" begin
        data1, data2 = rand(4, 4), rand(4, 4)
        block_sizes1, block_sizes2 = ([3, 1], [2, 2]), ([1, 3], [2, 2])
        block_array1 = BlockArray(data1, block_sizes1...)
        block_array2 = BlockArray(data2, block_sizes2...)

        tensor1 = Tensor(data1, [:i, :j])
        tensor2 = Tensor(data2, [:j, :k])
        block_tensor1 = Tensor(block_array1, [:i, :j])
        block_tensor2 = Tensor(block_array2, [:j, :k])

        contracted_tensor = contract(tensor1, tensor2)
        contracted_block_tensor = contract(block_tensor1, block_tensor2)

        @test parent(contracted_block_tensor) isa BlockArray
        @test contracted_block_tensor |> labels == (:i, :k)
        @test contracted_block_tensor |> blocksizes == ([3, 1], [2, 2])
        @test Array(parent(contracted_block_tensor)) ≈ parent(contracted_tensor)
    end

    @testset "contract block-unblocked tensors" begin
        data1, data2 = rand(4, 4), rand(4, 4)
        block_sizes = ([3, 1], [2, 2])
        block_array = BlockArray(data2, block_sizes...)

        tensor = Tensor(data1, [:i, :j])
        block_tensor = Tensor(block_array, [:j, :k])

        contracted_tensor = contract(tensor, block_tensor)

        @test contracted_tensor |> labels == (:i, :k)
        @test (contracted_tensor |> parent |> blocksizes)[2] == [2, 2]
        @test Array(parent(contracted_tensor)) ≈ parent(contract(tensor, Tensor(data2, [:j, :k])))
    end

    # It seems that svd is not yet supported for BlockArray:
    # https://github.com/JuliaArrays/BlockArrays.jl/issues/131
    # @testset "svd" begin
    #     data = rand(4, 4, 4)
    #     block_sizes = ([2, 2], [1, 3], [3, 1])
    #     block_array = BlockArray(data, block_sizes...)
    #     indices = (:i, :j, :k)

    #     tensor = Tensor(data, indices)
    #     block_tensor = Tensor(block_array, indices)

    #     U, S, V = svd(tensor; left_inds = (:i, :j))
    #     U̅, S̅, V̅ = svd(block_tensor; left_inds = (:i, :j))

    #     @test Array(parent(U̅)) ≈ parent(U)
    #     @test Array(parent(S̅)) ≈ parent(S)
    #     @test Array(parent(V̅)) ≈ parent(V)
    # end
end