#!/usr/bin/env python3

import tensorrt as trt
from infmoe import MoELayerConfig

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def create_moe_config_with_random_weight(filename, *args, **kwargs) -> MoELayerConfig:
    r"""
    Generate random MoE centroid & weight matrix
    """
    moe_config = MoELayerConfig(*args, **kwargs)
    moe_config.generate_random_centroids()
    moe_config.generate_random_weight_file(filename, False)
    return moe_config
