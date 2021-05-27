# moe-infer

Inference framework for MoE-based Transformer models, based on a TensorRT custom plugin
named `MoELayerPlugin` (including Python binding) that can run inference on a MoE layer with any sub-layer.

## Building

Dependencies:

* CUDA (>=10.2)
* cuDNN (>=8.0, corresponding to CUDA version)
* TensorRT (>=7.0, corresponding to CUDA & cuDNN version)
* zlib (to read `npz` files)
* meson & ninja (building system)

### Python (recommended)

To use TensorRT in Python, you need to first install:

* TensorRT pip package (see <https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-pip>)
* PyCUDA

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
* `hidden_size`: INT32, the intermediate size of feed forward network (might not be used by sub-layer)
* `max_concurrency`: INT32, maximal concurrent experts in GPU memory (default to 2), setting it too large will lead to OOM
* `expert_centroids`: FLOAT32 array, weight for dispatching tokens to experts, must be shape `(d_model, expert_count)` where `d_model` is the last dimension of input tensor (a.k.a. embedding size)
* `expert_weight_file`: null-terminated char pointer, path to expert weight file, to be read by implmentation of sub-layer
* `expert_sublayer_type`: type of sub-layer used, currently only `T5_FF` can be used

## Usage

The plugin can only handle MoE layers. To run inference with a full network, you should slice it before and after any MoE layer:

* for non-MoE layers, jsut save them as `onnx` / UFF format and use TensorRT to parse it;
* for MoE layers, dump expert centroids and weights of each expert separately (in the format mentioned below), manually initialize a layer using `MoELayerPlugin` with Python or C++.

Then you can concatenate MoE / non-MoE layers to obtain the full network, which can be later built into a TensorRT CUDA engine and used to run inference with / serialize & dump to file.

Python and C++ examples are given below for usage.

### Python

TBD.

### C++

TBD.

## Scheduling

TBD

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
