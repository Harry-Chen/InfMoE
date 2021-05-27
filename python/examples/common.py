#!/usr/bin/env python3

import tensorrt as trt
from trt_moe import MoELayerConfig

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def create_moe_config_with_random_weight() -> MoELayerConfig:
    r"""
    Generate random MoE centroid & weight matrix
    """
    # feel free to change the config here
    moe_config = MoELayerConfig(512, 20, 4096, 16384, 4, "T5_FF", 4, None, None)
    moe_config.generate_random_centroids()
    moe_config.generate_random_weight_file('/tmp/moe_weight.npz', False)
    return moe_config
