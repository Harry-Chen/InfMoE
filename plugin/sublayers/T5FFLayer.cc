#include "T5FFLayer.h"
#include "utility.h"

#include <string>
#include <cassert>

T5FFLayer::~T5FFLayer() {
}

bool T5FFLayer::init(const Dims *inputDims, int32_t nbInputs, const Dims *outputDims, int32_t nbOutputs) {
    // outputDims[0] should equal inputDims[0]
    assert(outputDims[0].nbDims == inputDims[0].nbDims && inputDims[0].nbDims == 1);
    auto &dim = inputDims[0];
    auto &dim2 = outputDims[0];
    assert(dim.nbDims == 2 && dim2.nbDims == 2);
    assert(dim.d[0] == dim2.d[0] && dim.d[1] == dim2.d[1]);
    mEmbeddingSize = dim.d[1];
    mSequenceLength = dim.d[0];
    return true;
}

Dims T5FFLayer::getOutputDimensions(int32_t index, const Dims* inputs, int32_t nbInputDims) {
    assert(index == 0);
    assert(nbInputDims == 3); // (batch_size, seq_len, embed_size)
    // output tensor should have the same shape with input tensor
    return inputs[0];
}

size_t T5FFLayer::weightSize() {
    // bias + weight
    auto layernorm_size = (size_t) mEmbeddingSize * sizeof(float) * 2;
    // d_model * d_ff * 3
    auto ff_size = (size_t) mEmbeddingSize * mHiddenSize * sizeof(float) * 3;
    return layernorm_size + ff_size;
}

size_t T5FFLayer::workspaceSize(int32_t maxBatchSize) {
    // TODO: calculate intermediate matrix size
    return 0;
}

void T5FFLayer::copyWeights(void *dst, int expert, cudaStream_t stream) {
    // TODO: copy weight of certain expert to dst
}

bool T5FFLayer::run(int32_t batchSize, const void *weights, const void *const *inputs, void **outputs, void *workspace,
                    cudaStream_t stream) {
    // TODO: run actual calculation
    // state = state + dense_relu_dense(layer_norm(state))
    // dense_relu_dense: hs = wo * (gelu(wi_0 * hs) * (wi_1 * hs))
}

void T5FFLayer::initialize() {
    assert(*mCublasHandle != nullptr);
    // load all weights to CPU memory (WARNING: huge memory consumption!)
    mSavedWeights = cnpy::npz_load(mWeightFile);
}

void T5FFLayer::terminate() {
    // free CPU memory
    delete mSavedWeights;
}
