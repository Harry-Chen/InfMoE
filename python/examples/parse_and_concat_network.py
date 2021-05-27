#!/usr/bin/env python3

import time
import numpy as np
import tensorrt as trt
import pycuda.autoinit # DO NOT REMOVE!

from common import TRT_LOGGER, create_moe_config_with_random_weight
from trt_moe import MoELayerPlugin, allocate_buffers, create_layer_from_plugin, do_inference


def parse_oonx_network(network, filename):
    onnx_parser = trt.OnnxParser(network, TRT_LOGGER)
    onnx_parser.parse_from_file(filename)
    assert onnx_parser.get_error(0) is None
    # show layers in onnx
    for i in range(24):
        layer = network.get_layer(i)
        print(layer.name, layer.type)
        input, output = layer.get_input(0), layer.get_output(0)
        print(input.shape if input is not None else 'NONE', output.shape if output is not None else 'NONE')


def create_moe_plugin() -> MoELayerPlugin:
    moe_config = create_moe_config_with_random_weight('/tmp/moe_weight_small.npz', 8, 2, 4, 8, 2, "T5_FF", 10)
    print(moe_config.__repr__())
    moe_plugin_class = MoELayerPlugin(moe_config)
    moe_plugin = moe_plugin_class.create_plugin()
    return moe_plugin_class, moe_plugin

def run_concatenated_network():

    # engine builder & config
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    config.max_workspace_size = (1 << 35)  # 16GB, change it if you have less GPU memory

    network = builder.create_network(flags=(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)))
    parse_oonx_network(network, '../../temp/naive_model.onnx')
    onnx_output = network.get_output(0)
    network.unmark_output(onnx_output)
    moe_plugin_py, moe_plugin = create_moe_plugin()
    moe_layer = create_layer_from_plugin(
        network, moe_plugin, [onnx_output], 'moe_1')
    network.mark_output(moe_layer.get_output(0))

    # optimzation profile (for dynamic shape support in ONNX layers)
    profile = builder.create_optimization_profile()
    layer_shape = (moe_plugin_py.config.max_batch_size, moe_plugin_py.config.seq_len, moe_plugin_py.config.embedding_size)
    profile.set_shape("input", layer_shape, layer_shape, layer_shape)
    config.add_optimization_profile(profile)

    # build engine
    print('Start building TensorRT engine, this might be slow...')
    engine = builder.build_engine(network, config)
    print('Done building engine')
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    # generate input
    layer_input = np.random.rand(*layer_shape).astype('f')
    np.copyto(inputs[0].host, layer_input.ravel())

    # run inference
    with engine.create_execution_context() as context:
        start = time.time()
        # note: because we have a explicit batch dimension, batch_size must be set to 1
        [layer_output] = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=1)
        end = time.time()
        elapsed = end - start
        print('Inference has cost', elapsed, 'seconds')
        layer_output = layer_output.reshape(*layer_shape)
        print('Output shape', layer_output.shape)


if __name__ == '__main__':
    run_concatenated_network()
