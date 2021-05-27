#!/usr/bin/env python3

import numpy as np
import os
import sys

from dataclasses import dataclass

@dataclass
class MoELayerConfig:
    r"""
    Configuration for MoE layer
    """
    
    seq_len: int
    expert_count: int
    embedding_size: int
    hidden_size: int
    max_concurrency: int
    sublayer_type: str
    max_batch_size: int
    expert_centroids: np.ndarray
    weight_file_path: str

    def generate_random_centroids(self) -> None:
        self.expert_centroids = np.random.rand(self.expert_count, self.embedding_size).astype('f')

    def generate_random_weight_file(self, weight_path: str, overwrite: bool):
        # check existence
        if os.path.isfile(weight_path):
            if overwrite:
                os.remove(weight_path)
            else:
                print(f'WARNING: weight file to generate exists: {weight_path}', file=sys.stderr)
                self.weight_file_path = weight_path
                return

        weights = {}
        print('Begin generate random weight for T5 FF Layer')
        layer_norm_weight = np.random.rand(self.embedding_size).astype('f')
        w_weight = np.random.rand(self.hidden_size, self.embedding_size).astype('f')
        print('End generate random weight')
        for i in range(self.expert_count):
            weights[f'{i}/layer_norm_weight'] = layer_norm_weight
            weights[f'{i}/wi_0_weight'] = w_weight
            weights[f'{i}/wi_1_weight'] = w_weight
            weights[f'{i}/wo_weight'] = w_weight.transpose()
        np.savez(weight_path, **weights)
        self.weight_file_path = weight_path
