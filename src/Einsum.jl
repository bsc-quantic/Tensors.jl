using LinearAlgebra

abstract type EinOperation{N} end
# abstract type PrimitiveEinOperation{N} <: EinOperation{N} end
# abstract type CompositeEinOperation{N} <: EinOperation{N} end

struct AxisSum <: EinOperation{1}
    inds::NTuple{N,Symbol} where {N}
end
AxisSum(inds...) = AxisSum(inds)

struct AxisPermutation <: EinOperation{1}
    perm::Dict{Symbol,Symbol}
end

struct Diagonal <: EinOperation{1}
    inds::NTuple{N,Symbol} where {N}
end
Diagonal(inds...) = new(inds)

struct HadamardProduct <: EinOperation{2}
    inds::NTuple{N,Symbol} where {N}
end
HadamardProduct(inds...) = new(inds)

struct OuterProduct <: EinOperation{2} end

struct TiledMatrixMultiplication <: EinOperation{2} end

function einsum end

function einsum(op::AxisSum, tensor::Tensor)
    i = findall(==(op), labels(tensor))
    data = dropdims(sum(parent(Tensor), dims=i), dims=i)
    labels = getindex((labels(tensor),), i)
    return Tensor(data, labels; tensor.meta...)
end

function einsum(::Diagonal, tensor::Tensor{T,2}) where {T}
    @assert allequal(labels(tensor))

    data = diag(parent(tensor))
    labels = first(labels(tensor))
    return Tensor(data, labels; tensor.meta...)
end