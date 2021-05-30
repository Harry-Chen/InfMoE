#!/usr/bin/env python3

import sys
import time
import numpy as np
import tensorrt as trt
import pycuda.autoinit # DO NOT REMOVE!

from common import TRT_LOGGER, create_moe_config_with_random_weight
from infmoe import MoELayerPlugin, allocate_buffers, create_layer_from_plugin, do_inference

def run_stacked_moe(moe_layers: int):
    r"""
    Create a network consisting only two MoE layers with same attributes & weights
    """

    # engine builder & config
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    config.max_workspace_size = (1 << 35)  # 16GB, change it if you have less GPU memory

    # moe plugin
    moe_config = create_moe_config_with_random_weight('/tmp/moe_weight.npz',
        seq_len=512, expert_count=20, embedding_size=4096, hidden_size=16384,
        max_concurrency=4, moe_variant="cpm_2", sublayer_type="T5_FF", max_batch_size=80,
        expert_centroids=None, weight_file_path=None
    )
    print(moe_config.__repr__())
    moe_plugin_class = MoELayerPlugin(moe_config)
    moe_plugin = moe_plugin_class.create_plugin()

    # network for builder
    layer_shape = (moe_config.max_batch_size, moe_config.seq_len, moe_config.embedding_size)
    network = builder.create_network(flags=(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)))
    input_layer = network.add_input(name="input_layer", dtype=trt.float32, shape=layer_shape)

    # stack MoE layers
    last_output = input_layer
    for l in range(moe_layers):
        moe_layer = create_layer_from_plugin(
            network, moe_plugin, [last_output], f'moe_{l+1}')
        last_output = moe_layer.get_output(0)
    network.mark_output(last_output)

    # build engine
    engine = builder.build_engine(network, config)
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
    if len(sys.argv) >= 2:
        moe_layers = int(sys.argv[1])
    else:
        moe_layers = 3

    run_stacked_moe(moe_layers)
