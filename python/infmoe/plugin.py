#!/usr/bin/env python3

import os
import sys
import ctypes
import tensorrt as trt
import numpy as np

from .config import MoELayerConfig


# load library from build directory first
# no directly insertion to sys.path to avoid pollution
def __search_library(lib_name):
    CURR_DIR = os.path.dirname(os.path.realpath(__file__))
    LIBRARY_SEARCH_PATH = [
        os.path.join(CURR_DIR, '../../plugin/builddir'),
        os.path.join(CURR_DIR, 'build'),
        CURR_DIR
    ]
    for d in LIBRARY_SEARCH_PATH:
        file_path = os.path.join(d, lib_name)
        if os.path.isfile(file_path):
            print(f'Loaded library {file_path}', file=sys.stderr)
            sys.stderr.flush()
            return file_path
    raise Exception(f'Cannot find libtrtmoelayer.so, have you run python3 setup.py build_ext?')


# global variables for TensorRT
TRT_MOE_PLUGIN_INFO = {
    'path': __search_library('libtrtmoelayer.so'),
    'name': 'MoELayerPlugin',
    'version': '1',
}
TRT_MOE_LAYER_LIB = ctypes.CDLL(TRT_MOE_PLUGIN_INFO['path'])
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')
TRT_REGISTRY = trt.get_plugin_registry()
TRT_MOE_LAYER_CREATOR = None


class MoELayerPlugin:
    r"""
    Python binding of MoELayerPlugin
    """

    __config: MoELayerConfig

    def __init__(self, config: MoELayerConfig, namespace: str = '') -> None:
        global TRT_MOE_LAYER_CREATOR
        # initialize creator
        if TRT_MOE_LAYER_CREATOR is None:
            TRT_MOE_LAYER_CREATOR = TRT_REGISTRY.get_plugin_creator(
                TRT_MOE_PLUGIN_INFO['name'], TRT_MOE_PLUGIN_INFO['version'], namespace)
            if TRT_MOE_LAYER_CREATOR is None:
                raise Exception(r'''
                Cannot initialize creator for MoELayerPlugin, check whether:
                1. libtrtmoelayer.so is correctly imported
                2. the TensorRT versions used by libtrtmoelayer.so and nvidia-tensorrt Python pacakge are the same
                ''')
        self.__config = config

    @property
    def config(self):
        return self.__config

    @config.setter
    def set_config(self, config: MoELayerConfig):
        self.__check_config(config)
        self.__config = MoELayerConfig

    def create_plugin(self, plugin_name='moe_layer_plugin'):
        attributes = self.__get_layer_attributes()
        return TRT_MOE_LAYER_CREATOR.create_plugin(plugin_name, field_collection=attributes)

    @classmethod
    def __check_config(config: MoELayerConfig):
        assert config.expert_count > 0
        assert config.seq_len > 0
        assert config.embedding_size > 0
        assert config.hidden_size > 0
        assert config.max_concurrency > 0
        assert config.max_batch_size > 0
        assert config.sublayer_type != ''
        assert config.expert_centroids.size > 0
        assert config.weight_file_path != ''
        assert config.moe_variant != ''
        # C++ plugin will do the other validity check
        pass

    def __get_layer_attributes(self) -> trt.PluginFieldCollection:

        if self.config is None:
            raise Exception('No MoE config provided')

        self.weight_file_path_encoded = self.config.weight_file_path.encode('utf-8')
        self.sublayer_type_encoded = self.config.sublayer_type.encode('utf-8')
        self.moe_variant_encoded = self.config.moe_variant.encode('utf-8')

        attributes = [
            trt.PluginField("expert_count", np.int32(
                self.config.expert_count), trt.PluginFieldType.INT32),
            trt.PluginField("embedding_size", np.int32(
                self.config.embedding_size), trt.PluginFieldType.INT32),
            trt.PluginField("hidden_size", np.int32(
                self.config.hidden_size), trt.PluginFieldType.INT32),
            trt.PluginField("max_concurrency", np.int32(
                self.config.max_concurrency), trt.PluginFieldType.INT32),
            trt.PluginField("expert_centroids", self.config.expert_centroids, trt.PluginFieldType.FLOAT32),
            trt.PluginField("expert_weight_file", self.weight_file_path_encoded, trt.PluginFieldType.UNKNOWN),
            trt.PluginField("expert_sublayer_type", self.sublayer_type_encoded, trt.PluginFieldType.UNKNOWN),
            trt.PluginField("moe_variant", self.moe_variant_encoded, trt.PluginFieldType.UNKNOWN),
        ]

        if self.config.layernorm_weight is not None:
            attributes.append(trt.PluginField("layernorm_weight", self.config.layernorm_weight, trt.PluginFieldType.FLOAT32))

        return trt.PluginFieldCollection(attributes)
