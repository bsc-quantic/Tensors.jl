module Tensors

include("Tensor.jl")
export Tensor
export labels, dim

include("Metadata.jl")
export tags, hastag, tag!, untag!

include("Numerics.jl")
export contract

include("Differentiation.jl")

end
