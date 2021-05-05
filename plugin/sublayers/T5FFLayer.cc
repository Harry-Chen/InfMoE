#include "T5FFLayer.h"

#include <cassert>

T5FFLayer::~T5FFLayer() {
    // TODO
}

bool T5FFLayer::init(const Dims *inputDims, int32_t nbInputs, const Dims *outputDims, int32_t nbOutputs) {
    // outputDims[0] should equal inputDims[0]
    assert(outputDims[0].nbDims == inputDims[0].nbDims && inputDims[0].nbDims == 1);
    auto &dim = inputDims[0];
    auto &dim2 = outputDims[0];
    // check and set dimensions: should be token * embedding
    assert(dim.nbDims == 2 && dim2.nbDims == 2);
    assert(dim.d[0] == dim2.d[0] && dim.d[1] == dim2.d[1]);
    mEmbeddingSize = dim.d[0];
    mSequenceLength = dim.d[1];
    return true;
}

size_t T5FFLayer::weightSize(int hiddenSize) {
    // TODO
}

size_t T5FFLayer::workspaceSize(int32_t maxBatchSize) {
    // TODO
}

bool T5FFLayer::run(int32_t batchSize, const void *weights, const void *const *inputs, void **outputs, void *workspace,
                    cudaStream_t stream, cublasHandle_t cublasHandle) {
    // TODO
}
