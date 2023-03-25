function contract(a::Tensor, b::Tensor, i=(∩(labels(a), labels(b))))
    ia = labels(a)
    ib = labels(b)
    i = ∩(i, ia, ib)

    ic = tuple(setdiff(ia ∪ ib, i isa Sequence ? i : [i])...)

    data = EinCode((String.(ia), String.(ib)), String.(ic))(a, b)

    # TODO merge metadata?
    return Tensor(data, ic)
end

contract(a, b) = a * b
contract(a::AbstractArray{T,0}, b) where {T} = contract(only(a), b)
contract(a, b::AbstractArray{T,0}) where {T} = contract(a, only(b))
contract(a::AbstractArray{<:Any,0}, b::AbstractArray{<:Any,0}) = contract(only(a), only(b))

function LinearAlgebra.svd(t::Tensor; left_inds=(), kwargs...)
    if any(∉(labels(t)), left_inds)
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
    Vt = reshape(Vt, size.((t,), right_inds)..., size(Vt, 2))

    vlind = Symbol(uuid4())
    vrind = Symbol(uuid4())

    U = Tensor(U, (left_inds..., vlind))
    s = Tensor(s, (vlind, vrind))
    Vt = Tensor(Vt, (vrind, right_inds...))

    return U, s, Vt
end
