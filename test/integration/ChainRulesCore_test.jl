@testset "ChainRulesCore" begin
    using Tensors: Tensor, contract
    using LinearAlgebra
    using ChainRulesCore: ProjectTo
    using ChainRulesTestUtils

    @testset "Tensor" begin
        test_frule(Tensor, rand(2, 2), (:i, :j), fkwargs = (; tags = Set(["TEST"])))
        test_rrule(Tensor, rand(2, 2), (:i, :j); fkwargs = (; tags = Set(["TEST"])))
    end

    @testset "ProjectTo" begin
        @testset "Matrix -> Matrix" begin
            data = rand(2, 2)

            tensor = Tensor(data, (:i, :j); tags = Set(["TEST"]))
            proj = ProjectTo(tensor)

            Δdata = rand(2, 2)
            Δ = proj(Δdata)
            @test parent(Δ) == Δdata
            @test labels(Δ) == labels(tensor)
            @test Δ.meta == tensor.meta
        end

        @testset "Matrix -> Diagonal" begin
            data = Diagonal(rand(2))

            tensor = Tensor(data, (:i, :j); tags = Set(["TEST"]))
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

    @testset "only" begin
        test_frule(only, Tensor(1.0))
        test_frule(only, Tensor(1.0))
    end

    @testset "contract" begin
        @testset "[Number-Number product]" begin
            @testset "T=Float" begin
                a = 5.0
                b = 2.0

                @test contract(a, b) == a * b
                test_rrule(contract, a, b)
                test_frule(contract, a, b)
            end

            @testset "T=Complex" begin
                a = 5.0 + 1.0im
                b = 2.0 - 2.0im

                @test contract(a, b) == a * b
                test_rrule(contract, a, b)
                test_frule(contract, a, b)
            end

            # TODO test two different eltypes
        end

        @testset "[Number-Tensor product]" begin
            @testset "T=Float" begin
                a = 5.0
                b = Tensor(rand(2, 2), (:i, :j))

                test_frule(contract, a, b)
                test_frule(contract, b, a)

                test_rrule(contract, a, b)
                test_rrule(contract, b, a)
            end

            @testset "T=Complex" begin
                a = 1.0 + 1im
                b = Tensor(rand(ComplexF64, 2, 2), (:i, :j))

                test_frule(contract, a, b)
                test_frule(contract, b, a)

                test_rrule(contract, a, b)
                test_rrule(contract, b, a)
            end

            # TODO test two different eltypes
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
