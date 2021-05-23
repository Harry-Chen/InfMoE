# moe-infer

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
layer_norm(hs) := wl * hs / sqrt(mean(pow(hs, 2)) + eps)
dense_relu_dense(hs) := (gelu(hs @ wi_0^T) * (hs @ wi_1^T)) @ wo^T
```

where `wi_0`, `wi_1` and `wo` are linear layers with no bias, first converting input tensor to 4 times large (in last dimension) then back.

The given `export_weight_file` must be a `npz` file containing the following variables (`n` varies from `0` to `expert_count - 1`): `n/layer_norm_weight`, `n/wi_0_weight`, `n/wi_1_weight`, `n/wo_weight`.


## Usage for Inference

The plugin can only handle MoE layers. To run inference with a full network, you should slice it before and after any MoE layer:

* for non-MoE layers, jsut save them as `onnx` format and use TensorRT to parse & run it;
* for MoE layers, dump expert centroids and weights of each expert separately (in the format mentioned above), manually initialize `MoELayerPlugin` with TensorRT API, then run it.

TODO: Python example

