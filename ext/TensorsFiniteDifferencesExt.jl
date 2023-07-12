module TensorsFiniteDifferencesExt

if isdefined(Base, :get_extension)
    using Tensors
else
    using ..Tensors
end

using FiniteDifferences: FiniteDifferences, to_vec

function FiniteDifferences.to_vec(tensor::Tensor)
    x_vec, from_vec = to_vec(parent(tensor))
    Tensor_from_vec(x_vec) = Tensor(from_vec(x_vec), labels(tensor); tensor.meta...)
    return x_vec, Tensor_from_vec
end

end
