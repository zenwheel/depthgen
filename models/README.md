---
tags:
- depth-estimation
library_name: coreml
license: apple-ascl
base_model:
  - apple/DepthPro
---

# DepthPro CoreML Models

DepthPro is a monocular depth estimation model. This means that it is trained to predict depth on a single image.

[DepthPro paper](https://arxiv.org/pdf/2410.02073)

[DepthPro original repo](https://huggingface.co/apple/DepthPro)

## Model Variants

| Variant                                                  | Size (MB) |
| ------------------------------------------------------- | ---------: |
|[DepthPro](DepthPro.mlpackage)| 1900 |
|[DepthPro: Pruned 10% Sparsity, Quantized Linear Symmetric](DepthProPruned10QuantizedLinear.mlpackage)| 1100 |
|[DepthPro Normalized Inverse Depth](DepthProNormalizedInverseDepth.mlpackage)| 1290 |
|[DepthPro Normalized Inverse Depth: Pruned 10% Sparsity, Quantized Linear Symmetric](DepthProNormalizedInverseDepthPruned10QuantizedLinear.mlpackage)| 745 |

## Model Inputs and Outputs

### DepthPro Normalized Inverse Depth Models

#### Inputs

- `image`: 1536x1536 3 color image.

#### Outputs

- `normalizedInverseDepth` 1536x1536 monochrome image.

### DepthPro Models

#### Inputs

- `image`: 1536x1536 3 color image.
- `originalWidth`: 1x1x1x1 Tensor containing the original width of the image before resizing.

#### Outputs

- `depthMeters`: 1x1x1536x1536 Tensor containing depth in meters.

## Download

Install `huggingface-cli`

```bash
brew install huggingface-cli
```

To download one of the `.mlpackage` folders to the `models` directory:

```bash
huggingface-cli download \
  --local-dir models --local-dir-use-symlinks False \
  KeighBee/coreml-DepthPro \
  --include "DepthProNormalizedInverseDepthPruned10QuantizedLinear.mlpackage/*" "DepthProPruned10QuantizedLinear.mlpackage/*"
```

To download everything, skip the `--include` argument.

## Integrate in Swift apps

The [`huggingface/coreml-examples`](https://github.com/huggingface/coreml-examples/blob/main/DepthProSample/README.md) repository contains sample Swift code for `DepthProNormalizedInverseDepthPruned10QuantizedLinear.mlpackage` and other models. See [the instructions there](https://github.com/huggingface/coreml-examples/tree/main/DepthProSample) to build the demo app, which shows how to use the model in your own Swift apps.