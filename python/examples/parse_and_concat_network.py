#!/usr/bin/env python3

import time
import numpy as np
import tensorrt as trt
import pycuda.autoinit # DO NOT REMOVE!

from .common import TRT_LOGGER, create_moe_config_with_random_weight
from trt_moe import MoELayerPlugin, allocate_buffers, create_layer_from_plugin, do_inference

def parse_oonx_network():
    pass

def create_moe_layer():
    pass

def run_concatenated_network():
    pass

if __name__ == '__main__':
    run_concatenated_network()
