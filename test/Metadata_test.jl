@testset "Metadata" begin
    @testset "tags" begin
        tensor =
            Tensor(zeros(2, 2, 2), (:i, :j, :k), tags = Set{String}(["TAG_A", "TAG_B"]))

        @test issetequal(tags(tensor), ["TAG_A", "TAG_B"])

        tag!(tensor, "TAG_C")
        @test hastag(tensor, "TAG_C")

        untag!(tensor, "TAG_C")
        @test !hastag(tensor, "TAG_C")

        @test untag!(tensor, "TAG_UNEXISTANT") == tags(tensor)
    end
end
