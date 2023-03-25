"""
    tags(tensor)

Return a set of `String`s associated to `tensor`.
"""
tags(t::Tensor) = t.meta[:tags]

"""
    tag!(tensor, tag)

Mark `tensor` with `tag`.
"""
tag!(t::Tensor, tag::String) = push!(tags(t), tag)

"""
    hastag(tensor, tag)

Return `true` if `tensor` contains tag `tag`.
"""
hastag(t::Tensor, tag::String) = tag ∈ tags(t)

"""
    untag!(tensor, tag)

Removes tag `tag` from `tensor` if present.
"""
untag!(t::Tensor, tag::String) = delete!(tags(t), tag)
