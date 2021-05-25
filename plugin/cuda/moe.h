#pragma once

#ifndef MOE_H
#define MOE_H

#include <cuda_runtime.h>

// select expert index & weight for each token
void moe_expert_select(
    const int token_num,
    const int token_len,
    const float *d_token_expert_aff,
    int *d_gate_selection,
    float *d_expert_weight,
    cudaStream_t stream
);

// count the tokens on each expert and obtain position for each token in routed_features
void moe_expert_count(
    const int token_num,
    const int expert_num,
    const int *d_gate_selection,
    int *d_token_pos,
    int *expert_offset,
    cudaStream_t stream
);

// scatter d_input & d_mix_coeff according to d_token_pos into d_routed_features
void moe_expert_scatter(
    const int token_num,
    const int token_len,
    const float *d_input,
    const float *d_mix_coeff,
    int *d_token_pos,
    float *d_routed_features,
    float *d_routed_mix_coeff,
    cudaStream_t stream
);

// gather d_routed_features back to d_output according to d_token_pos
void moe_expert_gather(
    const int token_num,
    const int token_len,
    const float *d_routed_features,
    const int *d_token_pos,
    float *d_output,
    cudaStream_t stream
);

// fused mix and gather kernel according to BASE Layer paper
// alpha = torch.sigmoid(routed_features.mv(self.expert_centroids[self.expert_id])).unsqueeze(1)
// routed_features = alpha * self.expert_network(routed_features) + (1 - alpha) * routed_features
// output = gather(routed_features, token_pos)
void moe_expert_base_layer_fused_mix_and_gather(
    const int token_num,
    const int token_len,
    const int *d_token_pos,
    const float *d_routed_features,
    const float *d_post_expert_features,
    float *d_mix_coeff,
    float *d_output,
    cudaStream_t stream
);

#endif // MOE_H
