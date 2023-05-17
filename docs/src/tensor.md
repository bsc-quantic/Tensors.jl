# The `Tensor` type

```@docs
Tensor
```

You can create a `Tensor` by passing an array and a list of `Symbol`s that name indices.

```@repl tensor
using Tensors # hide
T = Tensor(rand(3,5,2), (:i,:j,:k))
```

The _dimensionality_ or size of each index can be consulted using the `size` function.

```@docs
Base.size(::Tensor)
```

```@repl tensor
size(T)
size(T, :j)
length(T)
```

## Metadata

`Tensor`s may contain some metadata.

!!! warning "🚧 Work in progress 🚧"
    Currently there are only methods for accessing and modifying [Tags](@ref).

### Tags

```@docs
tags
tag!
hastag
untag!
```
