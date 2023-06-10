module TensorsChainRulesCoreExt

if isdefined(Base, :get_extension)
    using Tensors
else
    using ..Tensors
end

using ChainRulesCore

# projections
function ChainRulesCore.ProjectTo(tensor::T) where {T<:Tensor}
    ProjectTo{T}(;
        data=ProjectTo(parent(tensor)),
        labels=labels(tensor),
        meta=tensor.meta
    )
end

function (project::ProjectTo{Tensor{T,N,A}})(dx::Tensor{T,N,B}) where {T,N,A<:AbstractArray{T,N},B<:AbstractArray{T,N}}
    # TODO merge metadata?
    # TODO check if labels match?
    Tensor{T,N,A}(project.data(parent(dx)), project.labels; project.meta...)
end
function (project::ProjectTo{Tensor{T,N,A}})(dx::B) where {T,N,A<:AbstractArray{T,N},B<:AbstractArray{T,N}}
    Tensor{T,N,A}(project.data(dx), project.labels; project.meta...)
end
(project::ProjectTo{Tensor{T,N,A}})(dx::AbstractThunk) where {T,N,A<:AbstractArray{T,N}} = project(unthunk(dx))
(project::ProjectTo{T})(dx::Tensor{T,0}) where {T} = only(dx)

# constructor
function ChainRulesCore.frule((_, Δdata, _), f::Type{<:Tensor}, data, labels; meta...)
    tensor = f(data, labels; meta...)
    Δtensor = f(Δdata, labels; meta...)
    return tensor, Δtensor
end

function ChainRulesCore.rrule(f::Type{<:Tensor}, data, labels; meta...)
    t = f(data, labels; meta...)
    Tensor_pullback(t̄) = (NoTangent(), @thunk(parent(unthunk(t̄))), NoTangent())

    return t, Tensor_pullback
end

# only
function ChainRulesCore.frule((_, Δtensor), f::typeof(only), tensor::Tensor)
    result = f(tensor)
    Δresult = f(Δtensor)
    return result, Δresult
end

function ChainRulesCore.rrule(f::typeof(only), tensor::Tensor)
    result = f(tensor)
    only_pullback(Δtensor) = (NoTangent(), f(Δtensor))
    return result, only_pullback
end

# contract methods
function ChainRulesCore.frule((_, ȧ, ḃ)::NTuple{3,Any}, ::typeof(contract), a, b)
    c = contract(a, b)
    proj = ProjectTo(c)
    ċ = proj(contract(ȧ, b) + contract(a, ḃ))
    return c, ċ
end

# TODO try to integrate with above?
# TODO test
function ChainRulesCore.frule((_, ȧ, ḃ, _)::NTuple{4,Any}, ::typeof(contract), a, b, i)
    c = contract(a, b, i)
    proj = ProjectTo(c)
    ċ = proj(contract(ȧ, b, i) + contract(a, ḃ, i))
    return c, ċ
end

function ChainRulesCore.rrule(::typeof(contract), a::A, b::B) where {A,B}
    c = contract(a, b)
    project_a = ProjectTo(a)
    project_b = ProjectTo(b)
    project_c = ProjectTo(c)

    function contract_pullback(c̄)::Tuple{NoTangent,A,B} # TODO @thunk type inference
        ā = project_a(contract(project_c(unthunk(c̄)), b')) # TODO @thunk
        b̄ = project_b(contract(a', project_c(unthunk(c̄)))) # TODO @thunk

        return NoTangent(), ā, b̄
    end
    return c, contract_pullback
end

# TODO try to integrate with above?
# TODO test
function ChainRulesCore.rrule(::typeof(contract), a::A, b::B, i) where {A,B}
    c = contract(a, b, i)
    project_a = ProjectTo(a)
    project_b = ProjectTo(b)
    project_c = ProjectTo(c)

    function contract_pullback(c̄)::Tuple{NoTangent,A,B,NoTangent} # TODO @thunk type inference
        ā = project_a(contract(project_c(unthunk(c̄)), b', i)) # TODO @thunk
        b̄ = project_b(contract(a', project_c(unthunk(c̄)), i)) # TODO @thunk

        return NoTangent(), ā, b̄, NoTangent()
    end
    return c, contract_pullback
end

# function ChainRulesCore.rrule(::typeof(only), t::Tensor{T,0}) where {T}
#     data = only(t)

#     # TODO use `ProjectTo(t)`
#     only_pullback(d̄) = (NoTangent(), Tensor(fill(d̄), labels(t); t.meta...))
#     return data, only_pullback
# end

end