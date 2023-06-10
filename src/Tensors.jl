module Tensors

include("Tensor.jl")
export Tensor
export labels, dim

include("Metadata.jl")
export tags, hastag, tag!, untag!

include("Numerics.jl")
export contract

if !isdefined(Base, :get_extension)
    using Requires
end

@static if !isdefined(Base, :get_extension)
    function __init__()
        @require ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4" include("../ext/TensorsChainRulesCoreExt.jl")
    end
end

end
