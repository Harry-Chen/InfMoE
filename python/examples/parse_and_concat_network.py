#!/usr/bin/env python3

import os
import time
import numpy as np
import tensorrt as trt
import pycuda.autoinit  # DO NOT REMOVE!

from common import TRT_LOGGER, create_moe_config_with_random_weight
from infmoe import MoELayerPlugin, allocate_buffers, create_layer_from_plugin, do_inference


# contains a naive MLP network as described in 'generate_onnx.py'
DEMO_FILE_NAME = 'naive_model.onnx'


def parse_oonx_network(network, filename):
    onnx_parser = trt.OnnxParser(network, TRT_LOGGER)
    onnx_parser.parse_from_file(filename)
    assert onnx_parser.get_error(0) is None
    # show layers in onnx
    print('Layers in ONNX file:')
    i = 0
    while True:
        layer = network.get_layer(i)
        if layer is None:
            break
        print('Name:', layer.name, 'Type:', layer.type)
        input, output = layer.get_input(0), layer.get_output(0)
        print('Input:', input.shape if input is not None else 'NONE',
              'Output:', output.shape if output is not None else 'NONE')
        i += 1


def create_moe_plugin() -> MoELayerPlugin:
    moe_config = create_moe_config_with_random_weight('/tmp/moe_weight_small.npz',
        seq_len=8, expert_count=2, embedding_size=4, hidden_size=8,
        max_concurrency=2, moe_variant="cpm_2", sublayer_type="T5_FF", max_batch_size=10,
        expert_centroids=None, layernorm_weight=None, weight_file_path=None
    )
    print(moe_config.__repr__())
    moe_plugin_class = MoELayerPlugin(moe_config)
    moe_plugin = moe_plugin_class.create_plugin()
    return moe_plugin_class, moe_plugin


def run_concatenated_network():

    if not os.path.isfile(DEMO_FILE_NAME):
        raise Exception(
            f'{DEMO_FILE_NAME} not found, run generate_onnx.py to generate')

    # engine builder & config
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    # 16GB, change it if you have less GPU memory
    config.max_workspace_size = (1 << 35)

    # parse network from onnx
    network = builder.create_network(
        flags=(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)))
    parse_oonx_network(network, DEMO_FILE_NAME)
    # concat it with layer
    onnx_output = network.get_output(0)
    network.unmark_output(onnx_output)
    moe_plugin_py, moe_plugin = create_moe_plugin()
    moe_layer = create_layer_from_plugin(
        network, moe_plugin, [onnx_output], 'moe_1')
    network.mark_output(moe_layer.get_output(0))

    # optimzation profile (for dynamic shape support in ONNX layers)
    profile = builder.create_optimization_profile()
    layer_shape = (moe_plugin_py.config.max_batch_size,
                   moe_plugin_py.config.seq_len, moe_plugin_py.config.embedding_size)
    profile.set_shape('input', layer_shape, layer_shape, layer_shape)
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
        [layer_output] = do_inference(
            context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=1)
        end = time.time()
        elapsed = end - start
        print('Inference has cost', elapsed, 'seconds')
        layer_output = layer_output.reshape(*layer_shape)
        print('Output shape', layer_output.shape)


if __name__ == '__main__':
    run_concatenated_network()
