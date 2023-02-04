# FastTimm.jl

[![Build Status](https://github.com/lorenzoh/FastTimm.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/lorenzoh/FastTimm.jl/actions/workflows/CI.yml?query=branch%3Amain)

Use timm ([pytorch-image-models](https://github.com/rwightman/pytorch-image-models)) with FastAI.jl. FastTimm.jl integrates timm, the comprehensive library of pretrained computer vision models implemented in PyTorch with FastAI.jl. It allows you to load any timm model and train it using Julia packages.

To install FastTimm.jl, run the following in your Julia REPL:

```julia
# install Julia packages
using Pkg; pkg"add FastAI FastVision https://github.com/lorenzoh/FastTimm.jl https://github.com/lorenzoh/PyNNTraining.jl"
# install PyTorch and timm in PyCall's virtual environment. See https://pytorch.org/get-started/locally/ for other PyTorch installation options.
run(`$(PyCall.pyprogramname) -m pip install torch==1.13 torchvision==0.14 torchaudio==0.13 timm`)
```

timm models can be loaded through FastAI.jl's model registry:

```julia
using FastAI, FastTimm
model = load(models()["timm/resnet18"], pretrained=true)
```

To train a timm model, you need to

1. pass `model` to the `Learner`
2. pass the `ToPyTorch` callback (reexported from [PyNNTraining.jl](https://github.com/lorenzoh/PyNNTraining.jl)) to the `Learner`


```julia
data, blocks = load(datarecipes()["imagenette2-320"])
task = ImageClassificationSingle(blocks)
model = load(models()["timm/resnet18"], input=task.blocks.x, output=task.blocks.y)
learner = tasklearner(task, data; model=model, callbacks=[ToPyTorch("cuda")])
fitonecycle!(learner, 10)
```