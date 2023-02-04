
struct TimmBackboneVariant <: ModelVariant
    name::String
    nfeatures::Union{Int, Colon}
end

function loadvariant(variant::TimmBackboneVariant, xblock, yblock, checkpoint)
    # TODO: only take last feature maps
    # https://huggingface.co/docs/timm/feature_extraction#query-the-feature-information
    return timm.create_model(variant.name,
                          pretrained = checkpoint == "imagenet1k",
                          features_only=true)
end
compatibleblocks(variant::TimmBackboneVariant) =
    (FastVision.ImageTensor{2}(3), FastAI.ConvFeatures{2}(variant.nfeatures))


struct TimmClassifierVariant <: ModelVariant
    name::String
end

function loadvariant(variant::TimmClassifierVariant, xblock, yblock, checkpoint)
    m = timm.create_model(variant.name, pretrained = checkpoint == "imagenet1k")
end
compatibleblocks(::TimmClassifierVariant) = (FastVision.ImageTensor{2}(3), FastAI.OneHotTensor{0})


timm_variants(name) = [
        "classifier" => TimmClassifierVariant(name),
        "backbone" => TimmBackboneVariant(name, 512),  # TODO: use actual number of features
    ]


_timm_loadfn(name) = ckpt -> timm.create_model(name, pretrained = ckpt == "imagenet1k")

function register_models!(registry = FastAI.models())
    for name in timm.list_models(pretrained=true)
        id = "timm/$name"
        if !haskey(registry, id)
            push!(registry, (;
                id,
                variants=timm_variants(name),
                loadfn=_timm_loadfn(name),
                checkpoints=["imagenet1k"],
                backend=:pytorch))
        end
    end
end
