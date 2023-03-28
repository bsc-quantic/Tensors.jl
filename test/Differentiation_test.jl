@testset "ChainRules" begin
    using Tensors: Tensor, contract
    using LinearAlgebra
    using ChainRulesCore: ProjectTo

    using ChainRulesTestUtils
    ChainRulesTestUtils.ChainRulesCore.debug_mode() = true

    # fix for FiniteDifferences: `FiniteDifferences.to_vec` fails for Tensor{T,0}
    function ChainRulesTestUtils.FiniteDifferences.to_vec(x::T) where {T<:Tensor}
        x_vec, Array_from_vec = ChainRulesTestUtils.FiniteDifferences.to_vec(parent(x))
        _labels = labels(x)
        _meta = x.meta

        Tensor_from_vec(y) = T(Array_from_vec(y), _labels; _meta)

        return x_vec, Tensor_from_vec
    end

    @testset "Tensor" begin
        test_frule(Tensor, rand(2, 2), (:i, :j), fkwargs=(; tags=Set(["TEST"])))
        test_rrule(Tensor, rand(2, 2), (:i, :j); fkwargs=(; tags=Set(["TEST"])))
    end

    @testset "ProjectTo" begin
        @testset "Matrix -> Matrix" begin
            data = rand(2, 2)

            tensor = Tensor(data, (:i, :j); tags=Set(["TEST"]))
            proj = ProjectTo(tensor)

            Δdata = rand(2, 2)
            Δ = proj(Δdata)
            @test parent(Δ) == Δdata
            @test labels(Δ) == labels(tensor)
            @test Δ.meta == tensor.meta
        end

        @testset "Matrix -> Diagonal" begin
            data = Diagonal(rand(2))

            tensor = Tensor(data, (:i, :j); tags=Set(["TEST"]))
            proj = ProjectTo(tensor)

            Δdata = rand(2, 2)
            Δ = proj(Δdata)
            @test parent(Δ) == Diagonal(diag(Δdata))
            @test labels(Δ) == labels(tensor)
            @test Δ.meta == tensor.meta
        end

        # TODO fix these tests
        # @testset "Diagonal -> Matrix" begin
        #     data = Matrix(rand(2, 2))

        #     tensor = Tensor(data, (:i, :j); tags=Set(["TEST"]))
        #     proj = ProjectTo(tensor)

        #     Δdata = Diagonal(rand(2))
        #     Δ = proj(Δdata)
        #     @test parent(Δ) == Matrix(Δdata)
        #     @test labels(Δ) == labels(tensor)
        #     @test Δ.meta == tensor.meta
        # end
    end

    @testset "contract" begin
        @testset "[number-number product]" begin
            @testset "float" begin
                test_rrule(contract, 5.0, 2.0)
                test_frule(contract, 5.0, 2.0)
            end

            @testset "complex" begin
                test_rrule(contract, 5.0 + 1.0im, 2.0 - 2.0im)
                test_frule(contract, 5.0 + 1.0im, 2.0 - 2.0im)
            end
        end

        @testset "[number-tensor product]" begin
            b = Tensor(rand(2, 2), (:i, :j))
            z = 1.0 + 1im

            test_frule(contract, 5.0, b)
            test_rrule(contract, 5.0, b)

            # TODO fix these tests
            # test_frule(contract, z, b)
            # test_frule(contract, b, z)
            # test_rrule(contract, z, b)
            # test_rrule(contract, b, z)
        end

        @testset "[adjoint]" begin
            a = Tensor(rand(2, 2), (:i, :j))
            b = adjoint(a)

            test_frule(contract, a, b)
            test_rrule(contract, a, b)
        end

        # NOTE einsum: ij,ij->
        @testset "[inner product]" begin
            a = Tensor(rand(2, 2), (:i, :j))
            b = Tensor(rand(2, 2), (:i, :j))

            test_frule(contract, a, b)
            test_rrule(contract, a, b)
        end

        @testset "[outer product]" begin
            a = Tensor(rand(2), (:i,))
            b = Tensor(rand(2), (:j,))

            test_frule(contract, a, b)
            test_rrule(contract, a, b)
        end

        # NOTE einsum: ik,kj->ij
        @testset "[matrix multiplication]" begin
            @testset "[real numbers]" begin
                a = Tensor(rand(2, 2), (:i, :k))
                b = Tensor(rand(2, 2), (:k, :j))

                test_frule(contract, a, b)
                test_rrule(contract, a, b)
            end

            @testset "[complex numbers]" begin
                a = Tensor(rand(Complex{Float64}, 2, 2), (:i, :k))
                b = Tensor(rand(Complex{Float64}, 2, 2), (:k, :j))

                test_frule(contract, a, b)
                test_rrule(contract, a, b)
            end
        end
    end
end