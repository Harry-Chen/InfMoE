# mow-infer

Inference framework for MoE-based Transformer models, based on TensorRT a custom plugin
named `MoELayerPlugin` that can run inference on a MoE layer with any sub-layer.

## Building

Dependencies:

* CUDA (>=10.2)
* cuDNN (>=8.0, corresponding to CUDA version)
* TensorRT (>=7.0, corresponding to CUDA & cuDNN version)
* zlib (to read `npz` files)
* meson & ninja (building system)

Build commands:

```bash
cd plugin

# if you have cuDNN & TensorRT installed in search path
make builddir && make compile

# if you need to specify CUDA / cuDNN install location
# (CUDA can only be automatically searched by meson)
meson setup build -DWITH_TENSORRT=/path/to/tensorrt -DWITH_CUDNN=/path/to/cudnn
ninja -C builddir # or just run `make`
```

If everything goes well, you can find `libmoelayer.so` in `builddir`.

## Plugin parameters

When initializing `MoELayerPlugin` in TensorRT (either C++ or Python), the following parameters must be specified:

* `expert_count`: INT32, number of experts (sub-layers)
* `hidden_size`: INT32, the intermediate size of feed forward network (might not be used by sub-layer)
* `expert_centroids`: FLOAT32 array, weight for dispatching tokens to experts, must be shape `(expert_count, d_model)` where `d_model` is the last dimension of input tensor
* `expert_weight_file`: null-terminated char pointer, path to expert weight file, to be read by implmentation of sub-layer

## Scheduling

TBD

## Sub-layer

Extend `MoESubLayer` class to implement your own sub-layer.

### T5FFLayer

This project includes a feed-forward layer of T5 network defined as:

```text
hs := hs + dense_relu_dense(layer_norm(hs))
dense_relu_dense(hs) := wo * (gelu(wi_0 * hs) * (wi_1 * hs))
```

where `wi_0`, `wi_1` and `wo` are linear layers with no bias, first converting input tensor to 4 times large (in last dimension) then back.

The given `export_weight_file` must be a `npz` file containing the following variables (`n` varies from `0` to `expert_count - 1`):

* `n/layer_norm_weight`, `n/layer_norm_bias`
* `n/wi_0_weight`, `n/wi_1_weight`, `n/wo_weight`

