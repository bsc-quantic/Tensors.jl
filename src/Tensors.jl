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
        @require FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000" include(
            "../ext/TensorsFiniteDifferencesExt.jl",
        )
    end
end

end
