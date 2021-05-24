#pragma once

#ifndef MOE_H
#define MOE_H

#include <cuda_runtime.h>

void moe_expert_select(
    const int token_num,
    const int token_len,
    const float *d_token_expert_aff,
    int *d_gate_selection,
    cudaStream_t stream
);

void moe_expert_count(
    const int token_num,
    const int *d_gate_selection,
    int *d_token_pos,
    int *expert_count,
    int *expert_offset,
    cudaStream_t stream
);

void moe_expert_scatter(
    const int token_num,
    const int token_len,
    const float *d_token_expert_aff,
    int *d_token_pos,
    float *d_routed_features,
    cudaStream_t stream
);

// alpha = torch.sigmoid(routed_features.mv(self.expert_centroids[self.expert_id])).unsqueeze(1)
// routed_features = alpha * self.expert_network(routed_features) + (1 - alpha) * routed_features
void moe_expert_mix_base_layer(
    const int token_num,
    const int token_len,
    const float *d_expert_centroids,
    const float *d_post_expert_features,
    float *d_mix_coeff,
    float *d_routed_features,
    cudaStream_t stream
);

void moe_expert_gather(
    const int token_num,
    const int token_len,
    const float *d_routed_features,
    const int *d_token_pos,
    float *d_output,
    cudaStream_t stream
);

#endif // MOE_H
