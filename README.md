# InfMoE

Inference framework for MoE-based models, based on a TensorRT custom plugin
named `MoELayerPlugin` (including Python binding) that can run inference of MoE layers with any sub-layer on NVIDIA GPUs with minimal memory consumption.

InfMoE is open-sourced under [MIT License](LICENSE).

## Installation

Dependencies:

* CUDA (>=10.2)
* cuDNN (>=8.0, corresponding to CUDA version)
* TensorRT (>=7.0, corresponding to CUDA & cuDNN version)
* zlib (to read `npz` files)
* meson & ninja (building system)

### Python (recommended)

To use TensorRT in Python, you need to first install:

* TensorRT pip package (either from downloaded TensorRT package or from PyPI as `nvidia-tensorrt`, see <https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-pip>)
* PyCUDA

Simply you could run `python3 -m pip install -r requirements.txt`.

Note: If you install `nvidia-tensorrt` from PyPI (but not from downloaded TensorRT package), you **MUST** ensure the version of TensorRT that `MoELayerPlugin` links to matches the version that pip package uses (see `site-packages/tensorrt/`). Otherwise the plugin will not work correctly.

Then build this plugin:

```bash
cd python

# if you have cuDNN & TensorRT installed in search path, or
python3 setup.py build_ext
# if you need to specify CUDA / cuDNN install location
# (CUDA can only be automatically searched by meson)
python3 setup.py build_ext --tensorrt-prefix=/path/to/tensorrt --cudnn-prefix=/path/to/cudnn

python3 setup.py install .
```

You can also use `bdist_wheel` or other commands provided by `setuptools`. You can pass `--debug` to `build_ext` to enable verbose logging & keep the symbols for debugging purpose.

### C++ only (advanced)

```bash
cd plugin

# if you have cuDNN & TensorRT installed in search path
make builddir && make compile

# if you need to specify CUDA / cuDNN install location
# (CUDA can only be automatically searched by meson)
meson setup build -DWITH_TENSORRT=/path/to/tensorrt -DWITH_CUDNN=/path/to/cudnn
ninja -C builddir # or just run `make`
```

If everything goes well, you can find `libtrtmoelayer.so` in `builddir`. Similarly you can pass `-DDEBUG=true` to `meson setup` for debugging.

## Plugin attributes

When initializing `MoELayerPlugin` in TensorRT (either C++ or Python), the following attributes must be specified:

* `expert_count`: INT32, number of experts (sub-layers)
* `embedding_size`: INT32, the input & output size of expert network
* `hidden_size`: INT32, the intermediate size of feed forward network (might not be used by sub-layer)
* `max_concurrency`: INT32, maximal concurrent experts in GPU memory (default to 2), setting it too large will lead to OOM
* `expert_centroids`: FLOAT32 array, weight for dispatching tokens to experts, must be shape `(d_model, expert_count)` where `d_model` is the last dimension of input tensor (a.k.a. embedding size)
* `expert_weight_file`: null-terminated CHAR array, path to expert weight file, to be read by implmentation of sub-layer
* `expert_sublayer_type`: null-terminated CHAR array, type of sub-layer used, currently only `T5_FF` can be used
* `moe_variant`: null-terminated CHAR array, variant type of MoE layer, used to decide different behaviours (can be `cpm_2`, `base_layer` or `default`)
* `layernorm_weight`: FLOAT32 array, weight of layer norm layer applied to input before calculating expert affliation / score, must be provided when `moe_variant` is `cpm_2`

## Usage

Currently InfMoE can only handle MoE layers with FP32 parameters, input & output. To run inference with a full network, you should slice it before and after any MoE layer:

* For non-MoE layers, jsut save them as `onnx` / UFF format and use TensorRT to parse it into a network ([Python](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#import_onnx_python) / [C++](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#import_onnx_c)). Or you can use TensorRT API to construct the network manually ([Python](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#network_python) / [C++](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#create_network_c)).
* For MoE layers, dump expert centroids and weights of each expert separately (in the format mentioned below), create a layer using `MoELayerPlugin` with Python or C++ (see examples).

Then you can concatenate MoE / non-MoE layers to obtain the full network (or replace any specific 'placeholder' layer with MoE layer), which can be later built into a TensorRT CUDA engine and used to run inference with / serialize & dump to file.

We provide several Python examples in `python/examples` showing how to do the aforementioned work. You can run them after installing this plugin. You are encouraged to read [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html) to understand its workflow prior to using this plugin.

## Error handling

InfMoE requires that none of the following tensors contains `NaN` values:

* layer input
* expert centroids
* weight of layer norm (if applicable)

It will also check the shape and data type of all parameters, input & output tensors. If any misconfiguration is found, it will print error message to `stderr` and abort the whole process.

## Scheduling

See CPM-2 paper for scheduling details. To be ported to public source code soon.

## Sub-layer

We have provided some sublayers in `plugin/sublayers`. To implement your own sub-layer, you need to:

* Extend `MoESubLayer` class
* Add your layer name and initialization code to `MoELayerPlugin.h` (in `sublayer_type`) and `MoELayerPlugin.cc` (in `MoELayerPlugin::createSublayer()`)
* Add your source file (`.cpp` only) to `meson.build`
* Rebuild the plugin

### T5FFLayer (`T5_FF`)

This project includes an sublayer implementation of feed-forward layer in T5 network. It is defined as:

```text
hs := hs + dense_relu_dense(layer_norm(hs))
layer_norm(hs) := wl * hs / sqrt(mean(pow(hs, 2)) + eps)
dense_relu_dense(hs) := (gelu(hs @ wi_0^T) * (hs @ wi_1^T)) @ wo^T
```

where `wi_0`, `wi_1` and `wo` are linear layers with no bias, first converting input tensor to 4 times large (in last dimension) then back.

The given `export_weight_file` must be a `npz` file containing the following variables (`n` varies from `0` to `expert_count - 1`): `n/layer_norm_weight`, `n/wi_0_weight`, `n/wi_1_weight`, `n/wo_weight`.

### IdentityLayer (`Identity`)

This layer **DOES NOTHING** (thus use none of the provided plugin attributes), just copies the input directly to the output. It is intended for debugging purpose only.
