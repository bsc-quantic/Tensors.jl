using BlockArrays

struct BlockedTensor{T,N,A<:AbstractArray{T,N}} <: Tensor{T,N,A}
    data::BlockArray{T,N,A}
    labels::NTuple{N,Symbol}
    meta::Dict{Symbol,Any}

    function BlockedTensor{T,N,A}(data::A, labels::NTuple{N,Symbol}; meta...) where {T,N,A<:AbstractArray{T,N}}
        block_data = BlockArray(data) # Convert data to a BlockArray
        new{T,N,BlockArray{T,N,A}}(block_data, labels; meta...)
    end
end

# Convenience constructor
BlockedTensor(data, labels::Vector{Symbol}; meta...) = BlockedTensor(data, tuple(labels...); meta...)
BlockedTensor(data::A, labels::NTuple{N,Symbol}; meta...) where {T,N,A<:AbstractArray{T,N}} =
    BlockedTensor{T,N,A}(data, labels; meta...)

# Create BlockedTensor from Tensor
function to_blocked(t::Tensor)
    return BlockedTensor(parent(t), labels(t); t.meta...)
end

function contract(a::BlockedTensor, b::BlockedTensor; dims=(∩(labels(a), labels(b))))
    ia = labels(a)
    ib = labels(b)
    i = ∩(dims, ia, ib)

    ic = tuple(setdiff(ia ∪ ib, i isa Base.AbstractVecOrTuple ? i : (i,))...)

    data = EinCode((String.(ia), String.(ib)), String.(ic))(parent(a), parent(b))

    # TODO merge metadata?
    return BlockedTensor(data, ic)
end
