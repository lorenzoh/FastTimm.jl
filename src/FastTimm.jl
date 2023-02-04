module FastTimm

using PyCall
using PyNNTraining: totorch, ToPyTorch
using FastAI: FastAI
using FastAI.Registries: ModelVariant
using FastVision: FastVision, Image, ConvFeatures

const timm = PyCall.PyNULL()

include("variants.jl")
# TODO: add setup code
# TODO: add backbone variant
# TODO: add classifier variant
# TODO: add metadata from results CSV

function __init__()
    copy!(timm, pyimport("timm"))
    register_models!(FastAI.models())
end

export ToTorch

end
