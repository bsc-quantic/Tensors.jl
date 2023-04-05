using OMEinsum
using LinearAlgebra
using UUIDs: uuid4

"""
    contract(::Tensor, ::Tensor[, i])

Perform tensor contraction operation.
"""
function contract(a::Tensor, b::Tensor, i=(∩(labels(a), labels(b))))
    ia = labels(a)
    ib = labels(b)
    i = ∩(i, ia, ib)

    ic = tuple(setdiff(ia ∪ ib, i isa Base.AbstractVecOrTuple ? i : (i,))...)

    data = EinCode((String.(ia), String.(ib)), String.(ic))(parent(a), parent(b))

    # TODO merge metadata?
    return Tensor(data, ic)
end

contract(a::Union{T,AbstractArray{T,0}}, b::Tensor{T}) where {T} = contract(Tensor(a), b)
contract(a::Tensor{T}, b::Union{T,AbstractArray{T,0}}) where {T} = contract(a, Tensor(b))
contract(a::AbstractArray{<:Any,0}, b::AbstractArray{<:Any,0}) = contract(Tensor(a), Tensor(b)) |> only
contract(a::Number, b::Number) = contract(fill(a), fill(b))

"""
    *(::Tensor, ::Tensor)

Alias for [`contract`](@ref).
"""
Base.:*(a::Tensor, b::Tensor) = contract(a, b)
Base.:*(a::Tensor, b) = contract(a, b)
Base.:*(a, b::Tensor) = contract(a, b)

function LinearAlgebra.svd(t::Tensor, left_inds=(); kwargs...)

    if isempty(left_inds)
        throw(ErrorException("no left-indices in SVD factorization"))
    elseif any(∉(labels(t)), left_inds)
        # TODO better error exception and checks
        throw(ErrorException("all left-indices must be in $(labels(t))"))
    end

    right_inds = setdiff(labels(t), left_inds)
    if isempty(right_inds)
        # TODO better error exception and checks
        throw(ErrorException("no right-indices in SVD factorization"))
    end

    # permute array
    tensor = permutedims(t, (left_inds..., right_inds...))
    data = reshape(parent(tensor), prod(i -> size(t, i), left_inds), prod(i -> size(t, i), right_inds))

    # compute SVD
    U, s, Vt = svd(data; kwargs...)

    # tensorify results
    U = reshape(U, size.((t,), left_inds)..., size(U, 2))
    s = Diagonal(s)
    Vt = reshape(Vt', size.((t,), right_inds)..., size(Vt, 2))

    vlind = Symbol(uuid4())
    vrind = Symbol(uuid4())

    U = Tensor(U, (left_inds..., vlind))
    s = Tensor(s, (vlind, vrind))
    Vt = Tensor(Vt, (vrind, right_inds...))

    return U, s, Vt
end
