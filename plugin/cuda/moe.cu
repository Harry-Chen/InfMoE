#include "moe.h"

#include <cuda_runtime.h>

#include "../utility.h"
#include "../thirdparty/dbg.h"
#include "common.cuh"

namespace {

template <typename T, bool USE_WARP_SHFL>
__global__ void expert_select_top1_kernel(
    const int token_num,
    const int token_len,
    const T *token_expert_aff,
    int *gate_selection,
    T *expert_weight
) {
    int row_id = (blockIdx.x * blockDim.x + threadIdx.x);
    if (USE_WARP_SHFL) row_id /= WARP_SIZE;
    if (row_id > token_num) return;

    const T *row_ptr = token_expert_aff + token_len * row_id;
    T row_max = 0;
    int row_max_pos = -1;

    if constexpr (USE_WARP_SHFL) {
        const int warp_offset = threadIdx.x % WARP_SIZE;
        // use one warp for each row
        for (int i = 0; i < token_len; i += WARP_SIZE) {
            // obtain max of 32 numbers
            T data = 0; // default invalid value
            int token_offset = i + warp_offset;
            if (token_offset < token_len) data = row_ptr[token_offset];
            // warp shuffle for max & pos
            T local_max = data;
            int local_max_pos = token_offset;
    #pragma unroll
            for (int i = 16; i > 0; i >>= 1) {
                T higher_lane_max = __shfl_down_sync(0xFFFFFFFF, local_max, i);
                T higher_lane_max_pos = __shfl_down_sync(0xFFFFFFFF, local_max_pos, i);
                if (higher_lane_max > local_max) {
                    local_max = higher_lane_max;
                    local_max_pos = higher_lane_max_pos;
                }
            }
            if (warp_offset == 0) {
                if (local_max > row_max) {
                    row_max = local_max;
                    row_max_pos = local_max_pos;
                }
            }
        }
        if (warp_offset == 0) {
            gate_selection[row_id] = row_max_pos;
            if (expert_weight != nullptr) expert_weight[row_id] = row_max;
        }
    } else {
        // use one thread for each row
        for (int i = 0; i < token_len; ++i) {
            float data = row_ptr[i];
            if (data > row_max) {
                row_max = data;
                row_max_pos = i;
            }
        }
        gate_selection[row_id] = row_max_pos;
        if (expert_weight != nullptr) expert_weight[row_id] = row_max;
    }
}


// the following scatter / gather kernels are adapted from https://github.com/laekov/fastmoe/blob/v0.1.2/cuda/moe_compute_kernel.cu
template <typename T>
__global__ void batch_scatter_kernel(size_t wid, const int *pos, const T *inbuf, T *oubuf) { 
	inbuf += wid * pos[blockIdx.x];
	oubuf += wid * blockIdx.x;
	for (int i = threadIdx.x; i < wid; i += blockDim.x) {
		oubuf[i] = inbuf[i];
	}
}

template <typename T>
__global__ void batch_gather_kernel(size_t wid, const int *pos, const T *inbuf, T *oubuf) { 
	inbuf += wid * blockIdx.x;
	oubuf += wid * pos[blockIdx.x];
	for (int i = threadIdx.x; i < wid; i += blockDim.x) {
		oubuf[i] = inbuf[i];
	}
}

template <typename T, typename U>
__global__ void batch_scatter_feature_and_weight_kernel(
    size_t wid, const int *pos, const T *in_feat, T *out_feat, const U *in_weight, U *out_weight
) { 
	int token_offset = blockIdx.x;
    in_feat += wid * pos[token_offset];
	out_feat += wid * token_offset;
	// copy features
    for (int i = threadIdx.x; i < wid; i += blockDim.x) {
		out_feat[i] = in_feat[i];
	}
    // copy weight
    if (threadIdx.x == 0 && in_weight != nullptr) {
        out_weight[token_offset] = in_weight[pos[token_offset]];
    }
}

template <typename T, typename U>
__global__ void fused_batch_mix_and_gather_kernel(
    size_t token_num, const int *pos, const U *mix_coeff, const T *inbuf1, const T *inbuf2, T *oubuf
) { 
    int token_offset = blockIdx.x;
    U alpha = sigmoid(mix_coeff[token_offset]);
	inbuf1 += token_num * token_offset;
    inbuf2 += token_num * token_offset;
	oubuf += token_num * pos[token_offset];
	for (int i = threadIdx.x; i < token_num; i += blockDim.x) {
		oubuf[i] = alpha * inbuf1[i] + (1 - alpha) * inbuf2[i];
	}
}


}; // unnamed namespace

void moe_expert_select(
    const int token_num,
    const int token_len,
    const float *d_token_expert_aff,
    int *d_gate_selection,
    float *d_expert_weight,
    cudaStream_t stream
) {
    dbg("before kernel");
    expert_select_top1_kernel<float, false><<<ceiling(token_num, 512), 512, 0, stream>>>(
        token_num, token_len, d_token_expert_aff, d_gate_selection, d_expert_weight
    );
    dbg("after kernel");
    CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
    dbg("after sync");
}

void moe_expert_count(
    const int token_num,
    const int expert_num,
    const int *d_gate_selection,
    int *d_token_pos,
    int *expert_offset,
    cudaStream_t stream
) {
    auto gate_selection = new int[token_num];
    auto token_pos = new int[token_num];
    auto expert_count = new int[expert_num];

    dbg("before copy");
    CUDA_SAFE_CALL(cudaMemcpyAsync(
        gate_selection, d_gate_selection, token_num * sizeof(int), cudaMemcpyDeviceToHost, stream
    ));
    CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
    dbg("after copy");

    dbg("before sort");
    // run counting sorting on CPU
    for (int i = 0; i < token_num; ++i) {
        expert_count[gate_selection[i]]++;
    }
    expert_offset[0] = 0;
    for (int i = 1; i < expert_num; ++i) {
        expert_offset[i] = expert_offset[i - 1] + expert_count[i - 1];
    }
    // use expert_count to fill
    memcpy(expert_count, expert_offset, sizeof(int) * expert_num);
    for (int i = 0; i < token_num; ++i) {
        token_pos[expert_count[gate_selection[i]]++] = i;
    }
    dbg("after sort");

    // copy back to GPU
    CUDA_SAFE_CALL(cudaMemcpyAsync(d_token_pos, token_pos, token_num * sizeof(int), cudaMemcpyHostToDevice, stream));
    delete[] gate_selection;
    delete[] expert_count;
    CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
    delete[] token_pos;
}

void moe_expert_scatter(
    const int token_num,
    const int token_len,
    const float *d_input,
    const float *d_mix_coeff,
    int *d_token_pos,
    float *d_routed_features,
    float *d_routed_mix_coeff,
    cudaStream_t stream
) {
    batch_scatter_feature_and_weight_kernel<<<token_num, 256, 0, stream>>>(
        token_len, d_token_pos, d_input, d_routed_features, d_mix_coeff, d_routed_mix_coeff
    );
    CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
}

void moe_expert_gather(
    const int token_num,
    const int token_len,
    const float *d_routed_features,
    const int *d_token_pos,
    float *d_output,
    cudaStream_t stream
) {
    batch_gather_kernel<<<token_num, 256, 0, stream>>>(token_len, d_token_pos, d_routed_features, d_output);
    CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
}

void moe_expert_base_layer_fused_mix_and_gather(
    const int token_num,
    const int token_len,
    const int *d_token_pos,
    const float *d_routed_features,
    const float *d_post_expert_features,
    float *d_mix_coeff,
    float *d_output,
    cudaStream_t stream
) {
    // routed_features = alpha * self.expert_network(routed_features) + (1 - alpha) * routed_features
    // output = gather(routed_features, token_pos)
    fused_batch_mix_and_gather_kernel<<<token_num, 256, 0, stream>>>(
        token_len, d_token_pos, d_mix_coeff, d_post_expert_features, d_routed_features, d_output
    );
    CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
}
