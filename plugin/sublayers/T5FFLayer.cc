#include "T5FFLayer.h"

#include <cublas_v2.h>
#include <NvInferPlugin.h>

#include <cassert>
#include <string>

#include "../thirdparty/dbg.h"
#include "../cuda/ops.h"
#include "../utility.h"

using namespace nvinfer1;

T5FFLayer::~T5FFLayer() {
    this->terminate();
    dbg("destructing T5FFLayer");
}

bool T5FFLayer::configureWithFormat(const Dims *inputDims, int32_t nbInputs, const Dims *outputDims, int32_t nbOutputs) {
    assert(nbInputs == 1 && nbOutputs == 1);
    // outputDims[0] should equal inputDims[0]
    // dbg(outputDims[0].nbDims, inputDims[0].nbDims);
    assert(outputDims[0].nbDims == inputDims[0].nbDims && inputDims[0].nbDims == 3);
    auto &dim = inputDims[0];
    auto &dim2 = outputDims[0];
    assert(dim.d[2] == dim2.d[2] && dim.d[1] == dim2.d[1] && dim.d[0] == dim2.d[0]);
    mEmbeddingSize = dim.d[2];
    mSequenceLength = dim.d[1];
    // get CUDA device props
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&mDeviceProp, 0));
    assert(mDeviceProp.major >= 6); // we don't want too old devices
    return true;
}

DimsExprs T5FFLayer::getOutputDimensions(const DimsExprs* inputs, IExprBuilder& exprBuilder) {
    // output tensor should have the same shape with input tensor
    dbg(inputs[0].nbDims);
    return DimsExprs(inputs[0]);
}

size_t T5FFLayer::weightSize() { return layernormWeightSize() + 3 * intermediateFFWeightSize(); }

size_t T5FFLayer::workspaceSize(int32_t tokenCount) {
    dbg("call workspaceSize");
    // calculate intermediate matrix size for given count of tokens
    // all intermediate variables:
    // layernorm_output: token_num * d_ff
    // wi_0_o: token_num * hidden_size (normally 4 * d_ff)
    // wi_0_i: token_num * hidden_size
    return layernormOutputSize(tokenCount) + 2 * intermediateFFOutputSize(tokenCount);
}

void T5FFLayer::copyWeights(void *dst, int expert, cudaStream_t stream) {
    // dbg(expert);
    // copy weight of specified expert to dst
    auto weight_ptr_byte = static_cast<char *>(dst);

    // layernorm_weight: token_num
    auto layernorm_weight_raw = (*mSavedWeights)[std::to_string(expert) + "/layer_norm_weight"];
    assert(layernorm_weight_raw.num_bytes() == layernormWeightSize());
    auto *layernorm_weight = reinterpret_cast<float *>(weight_ptr_byte);
    CUDA_SAFE_CALL(cudaMemcpyAsync(layernorm_weight, layernorm_weight_raw.data<float>(), layernormWeightSize(),
                                   cudaMemcpyHostToDevice, stream));

    // wi_0_weight: hidden_size * d_model
    auto wi_0_weight_raw = (*mSavedWeights)[std::to_string(expert) + "/wi_0_weight"];
    assert(wi_0_weight_raw.num_bytes() == intermediateFFWeightSize());
    auto *wi_0_weight = reinterpret_cast<float *>(weight_ptr_byte + layernormWeightSize());
    CUDA_SAFE_CALL(cudaMemcpyAsync(wi_0_weight, wi_0_weight_raw.data<float>(), intermediateFFWeightSize(),
                                   cudaMemcpyHostToDevice, stream));

    // wi_1_weight: hidden_size * d_model
    auto wi_1_weight_raw = (*mSavedWeights)[std::to_string(expert) + "/wi_1_weight"];
    assert(wi_1_weight_raw.num_bytes() == intermediateFFWeightSize());
    auto *wi_1_weight = reinterpret_cast<float *>(weight_ptr_byte + layernormWeightSize() + intermediateFFWeightSize());
    CUDA_SAFE_CALL(cudaMemcpyAsync(wi_1_weight, wi_1_weight_raw.data<float>(), intermediateFFWeightSize(),
                                   cudaMemcpyHostToDevice, stream));

    // wo_weight: d_model * hidden_size
    auto wo_weight_raw = (*mSavedWeights)[std::to_string(expert) + "/wo_weight"];
    assert(wo_weight_raw.num_bytes() == intermediateFFWeightSize());
    auto *wo_weight =
        reinterpret_cast<float *>(weight_ptr_byte + layernormWeightSize() + intermediateFFWeightSize() * 2);
    CUDA_SAFE_CALL(
        cudaMemcpyAsync(wo_weight, wo_weight_raw.data<float>(), intermediateFFWeightSize(), cudaMemcpyHostToDevice, stream));
}

bool T5FFLayer::run(int32_t tokenCount, const void *weights, const void *input, void *output, void *workspace,
                    cudaStream_t stream) {

    assert(mCublasHandle != nullptr);
    // run actual calculation: hs := hs + dense_relu_dense(layer_norm(hs))
    auto workspace_ptr_byte = static_cast<char *>(workspace);
    auto weight_ptr_byte = static_cast<const char *>(weights);

    // layer_norm(hs) := wl * (hs / sqrt(mean(pow(hs, 2)) + eps))
    auto *expert_input = static_cast<const float *>(input);
    auto *layernorm_weight = reinterpret_cast<const float *>(weight_ptr_byte);
    auto *layernorm_output = reinterpret_cast<float *>(workspace_ptr_byte);
    layernorm_gpu<float, float>(layernorm_output, expert_input, tokenCount, mEmbeddingSize, (double)1e-6,
                                   layernorm_weight, nullptr, mDeviceProp.maxGridSize[1], stream);

    // dense_relu_dense(hs) := (gelu(hs @ wi_0^T) * (hs @ wi_1^T)) @ wo^T
    // TODO: maybe use cublasSgemmBatched for higher throughput
    float alpha = 1.0f, beta = 0.0f;
    // NOTE: cuBLAS is column major, and PyTorch linear layer requires y = x @ A^T, where y, x, A are all row major
    // considering y^T = A @ x^T, thus we just use y = cublasSgemm(A^T, x) for expected result

    // wi_0_o = ln_output @ wi_0^T
    auto *wi_0_weight = reinterpret_cast<const float *>(weight_ptr_byte + layernormWeightSize());
    auto *wi_0_output = reinterpret_cast<float *>(workspace_ptr_byte + layernormOutputSize(tokenCount));
    CUBLAS_SAFE_CALL(cublasSgemm_v2(mCublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, mHiddenSize, tokenCount, mEmbeddingSize,
                                    &alpha, wi_0_weight, mEmbeddingSize, layernorm_output, mEmbeddingSize, &beta,
                                    wi_0_output, mHiddenSize));
    // wi_1_o = ln_output @ wi_1^T
    auto *wi_1_weight =
        reinterpret_cast<const float *>(weight_ptr_byte + layernormWeightSize() + intermediateFFWeightSize());
    auto *wi_1_output = reinterpret_cast<float *>(workspace_ptr_byte + layernormOutputSize(tokenCount) +
                                                  intermediateFFOutputSize(tokenCount));
    CUBLAS_SAFE_CALL(cublasSgemm_v2(mCublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, mHiddenSize, tokenCount, mEmbeddingSize,
                                    &alpha, wi_1_weight, mEmbeddingSize, layernorm_output, mEmbeddingSize, &beta,
                                    wi_1_output, mHiddenSize));
    // wi_1_o = gelu(wi_0_o) * wi_1_o
    fused_gelu_dot_gpu(wi_0_output, wi_1_output, tokenCount * mHiddenSize, stream);
    CUDA_SAFE_CALL(cudaGetLastError());
    // copy input -> output
    // output = output + wi_1_o @ wo^T
    auto *wo_weight =
        reinterpret_cast<const float *>(weight_ptr_byte + layernormWeightSize() + intermediateFFWeightSize() * 2);
    auto *expert_output = reinterpret_cast<float *>(output);
    beta = 1.0f;
    CUDA_SAFE_CALL(cudaMemcpyAsync(output, input, sizeof(float) * tokenCount * mEmbeddingSize, cudaMemcpyDeviceToDevice, stream));
    CUBLAS_SAFE_CALL(cublasSgemm_v2(mCublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, mEmbeddingSize, tokenCount, mHiddenSize,
                                    &alpha, wo_weight, mHiddenSize, wi_1_output, mHiddenSize, &beta, expert_output,
                                    mEmbeddingSize));

    return true;
}

void T5FFLayer::initialize() {
    // load all weights to CPU memory (WARNING: huge memory consumption!)
    if (mSavedWeights == nullptr) mSavedWeights = cnpy::npz_load(mWeightFile);
    dbg("weights loaded");
}

void T5FFLayer::terminate() {
    dbg("call terminate");
    // free CPU memory
    if (mSavedWeights != nullptr) {
        delete mSavedWeights;
        mSavedWeights = nullptr;
    }
}
