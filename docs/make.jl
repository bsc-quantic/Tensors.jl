using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

push!(LOAD_PATH, "$(@__DIR__)/..")

using Documenter
using Tensors

DocMeta.setdocmeta!(Tensors, :DocTestSetup, :(using Tensors); recursive=true)

makedocs(
    modules=[Tensors],
    sitename="Tensors.jl",
    authors="Sergio Sánchez Ramírez and contributors",
    pages=Any[
        "Home"=>"index.md",
        "The `Tensor` type"=>"tensor.md",
        "Einstein Summation"=>"einsum.md",
    ],
    format=Documenter.HTML(; assets=["assets/images.css"]),
)
