#!/usr/bin/env python3

import time
import numpy as np
import tensorrt as trt
import pycuda.autoinit # DO NOT REMOVE!

from .common import TRT_LOGGER, create_moe_config_with_random_weight
from trt_moe import MoELayerPlugin, allocate_buffers, create_layer_from_plugin, do_inference

def run_two_layer_moe():
    r"""
    Create a network consisting only two MoE layers with same attributes & weights
    """

    # engine builder & config
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    config.max_workspace_size = (1 << 35)  # 16GB, change it if you have less GPU memory

    # moe plugin
    moe_config = create_moe_config_with_random_weight()
    print(moe_config.__repr__())
    moe_plugin_class = MoELayerPlugin(moe_config)
    moe_plugin = moe_plugin_class.create_plugin()

    # network for builder
    builder.max_batch_size = moe_config.max_batch_size
    network = builder.create_network()
    input_layer = network.add_input(name="input_layer", dtype=trt.float32, shape=(
        moe_config.seq_len, moe_config.embedding_size))
    # two stacked layers of MoE
    moe_layer = create_layer_from_plugin(
        network, moe_plugin, [input_layer], 'moe_1')
    moe_layer_2 = create_layer_from_plugin(
        network, moe_plugin, [moe_layer.get_output(0)], 'moe_2')
    network.mark_output(moe_layer_2.get_output(0))

    # build engine
    engine = builder.build_engine(network, config)
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    # generate input
    layer_shape = (moe_config.max_batch_size, moe_config.seq_len, moe_config.embedding_size)
    layer_input = np.random.rand(*layer_shape).astype('f')
    np.copyto(inputs[0].host, layer_input.ravel())

    # run inference
    with engine.create_execution_context() as context:
        start = time.time()
        [layer_output] = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=moe_config.max_batch_size)
        end = time.time()
        elapsed = end - start
        print('Inference has cost', elapsed, 'seconds')
        layer_output = layer_output.reshape(*layer_shape)
        print('Output shape', layer_output.shape)


if __name__ == '__main__':
    run_two_layer_moe()