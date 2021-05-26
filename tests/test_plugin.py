#!/usr/bin/env python3

import pytest
import sys
import os
import ctypes
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from dataclasses import dataclass

MOE_LAYER_PLUGIN_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../plugin/builddir/libtrtmoelayer.so'
)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

@dataclass
class MoEConfig:
    seq_len: int
    expert_count: int
    embedding_size: int
    hidden_size: int
    max_concurrency: int
    sublayer_type: str
    max_batch_size: int

    def generate_centroids(self) -> np.ndarray:
        return np.random.rand(self.expert_count, self.embedding_size).astype('f')

    def generate_weight_file(self) -> str:
        weight_path = '/tmp/moe_weight.npz'
        if os.path.isfile(weight_path):
            return weight_path
            # os.remove(weight_path)
        weights = {}
        print('Begin generate random weight')
        layer_norm_weight = np.random.rand(self.embedding_size).astype('f')
        w_weight = np.random.rand(self.hidden_size, self.embedding_size).astype('f')
        print('End generate random weight')
        for i in range(self.expert_count):
            weights[f'{i}/layer_norm_weight'] = layer_norm_weight
            weights[f'{i}/wi_0_weight'] = w_weight
            weights[f'{i}/wi_1_weight'] = w_weight
            weights[f'{i}/wo_weight'] = w_weight.transpose()
        np.savez(weight_path, **weights)
        return weight_path


def generate_attributes(config: MoEConfig, centroids, weight_file):
    # print(centroids.data, hex(centroids.__array_interface__['data'][0]))
    return trt.PluginFieldCollection([
        trt.PluginField("expert_count", np.int32(config.expert_count), trt.PluginFieldType.INT32),
        trt.PluginField("hidden_size", np.int32(config.hidden_size), trt.PluginFieldType.INT32),
        trt.PluginField("max_concurrency", np.int32(config.max_concurrency), trt.PluginFieldType.INT32),
        trt.PluginField("expert_centroids", centroids, trt.PluginFieldType.FLOAT32),
        trt.PluginField("expert_weight_file", weight_file.encode('utf-8'), trt.PluginFieldType.UNKNOWN),
        trt.PluginField("expert_sublayer_type", config.sublayer_type.encode('utf-8'), trt.PluginFieldType.UNKNOWN),
    ])


def create_moe_layer_plugin(config: MoEConfig):
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")
    ctypes.CDLL(MOE_LAYER_PLUGIN_PATH)
    registry = trt.get_plugin_registry()
    creator = registry.get_plugin_creator("MoELayerPlugin", "1", "")

    centroids = config.generate_centroids()
    weight_file = config.generate_weight_file()
    # print(centroids)
    field_collection = generate_attributes(config, centroids, weight_file)
    plugin = creator.create_plugin("", field_collection=field_collection)
    return plugin


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # print('before exec')
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    # print('after exec')
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

def test_moe_layer_plugin():

    # engine builder & config
    builder = trt.Builder(TRT_LOGGER) 
    config = builder.create_builder_config()
    config.max_workspace_size = int((1 << 35) * 1.5) # 16GB

    # network for builder
    network = builder.create_network()
    moe_config = MoEConfig(512, 20, 4096, 16384, 2, "T5_FF", 100)
    moe_plugin = create_moe_layer_plugin(moe_config)
    input_layer = network.add_input(name="input_layer", dtype=trt.float32, shape=(moe_config.seq_len, moe_config.embedding_size))
    moe_layer = network.add_plugin_v2(inputs=[input_layer], plugin=moe_plugin)
    moe_layer_2 = network.add_plugin_v2(inputs=[moe_layer.get_output(0)], plugin=moe_plugin)
    moe_layer.get_output(0).name = "moe_1_output"
    moe_layer_2.get_output(0).name = "moe_2_output"
    network.mark_output(moe_layer_2.get_output(0))
    builder.max_batch_size = moe_config.max_batch_size

    # build engine
    engine = builder.build_engine(network, config)
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    with engine.create_execution_context() as context:
        layer_input = np.random.rand(moe_config.max_batch_size, moe_config.seq_len, moe_config.embedding_size)
        # print(layer_input @ centroids.transpose())
        # print(layer_input)
        np.copyto(inputs[0].host, layer_input.ravel())
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        stream.synchronize()
        start = time.time()
        for i in range(1):
            print('inference', i)
            context.execute_async(batch_size=moe_config.max_batch_size, bindings=bindings, stream_handle=stream.handle)
            # [_] = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=moe_config.max_batch_size)
        # print(layer_output)
        end = time.time()
        print((end - start) / 10 / moe_config.max_batch_size)
        stream.synchronize()
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]

if __name__ == '__main__':
    test_moe_layer_plugin()
