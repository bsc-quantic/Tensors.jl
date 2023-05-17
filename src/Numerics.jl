using OMEinsum
using LinearAlgebra
using UUIDs: uuid4

"""
    contract(::Tensor, ::Tensor[, i])

Perform tensor contraction operation.
"""
function contract(a::Tensor, b::Tensor; dims=(∩(labels(a), labels(b))))
    ia = labels(a)
    ib = labels(b)
    i = ∩(dims, ia, ib)

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

LinearAlgebra.svd(t::Tensor; left_inds=(), kwargs...) = svd(t, left_inds; kwargs...)

function LinearAlgebra.svd(t::Tensor, left_inds; kwargs...)
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
    U, s, V = svd(data; kwargs...)

    # tensorify results
    U = reshape(U, ([size(t, ind) for ind in left_inds]..., size(U, 2)))
    s = Diagonal(s)
    Vt = reshape(V', (size(V', 1), [size(t, ind) for ind in right_inds]...))

    vlind = Symbol(uuid4())
    vrind = Symbol(uuid4())

    U = Tensor(U, (left_inds..., vlind))
    s = Tensor(s, (vlind, vrind))
    Vt = Tensor(Vt, (vrind, right_inds...))

    return U, s, Vt
end

LinearAlgebra.qr(t::Tensor; left_inds=(), kwargs...) = qr(t, left_inds; kwargs...)

function LinearAlgebra.qr(t::Tensor, left_inds; virtualind::Symbol=Symbol(uuid4()), kwargs...)
    # TODO better error exception and checks
    isempty(left_inds) && throw(ErrorException("no left-indices in QR factorization"))
    left_inds ⊆ labels(t) || throw(ErrorException("all left-indices must be in $(labels(t))"))

    right_inds = setdiff(labels(t), left_inds)
    isempty(right_inds) && throw(ErrorException("no right-indices in QR factorization"))

    # permute array
    tensor = permutedims(t, (left_inds..., right_inds...))
    data = reshape(parent(tensor), prod(i -> size(t, i), left_inds), prod(i -> size(t, i), right_inds))

    # compute QR
    Q, R = qr(data; kwargs...)

    # tensorify results
    Q = reshape(Q, ([size(t, ind) for ind in left_inds]..., size(Q, 2)))
    R = reshape(R, (size(R, 1), [size(t, ind) for ind in right_inds]...))

    Q = Tensor(Q, (left_inds..., virtualind))
    R = Tensor(R, (virtualind, right_inds...))

    return Q, R
end
