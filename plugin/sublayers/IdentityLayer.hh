#pragma once

#ifndef IDENTITYLAYER_H
#define IDENTITYLAYER_H

#include <NvInferPlugin.h>
#include <cuda_runtime.h>

#include "../thirdparty/dbg.h"
#include "SubLayer.h"

class IdentityLayer : public MoESubLayer {

   public:
    explicit IdentityLayer() : MoESubLayer(0, 0, 0, nullptr, 0) {}
    virtual ~IdentityLayer() override {}
    virtual bool configureWithFormat(const Dims *inputDims, [[maybe_unused]] int32_t nbInputs, const Dims *outputDims,
                                     [[maybe_unused]] int32_t nbOutputs) override {
        assert(outputDims[0].nbDims == inputDims[0].nbDims && inputDims[0].nbDims == 3);
        auto &dim = inputDims[0];
        mEmbeddingSize = dim.d[2];
        return true;
    }
    virtual size_t weightSize() override { return 0; }
    virtual size_t workspaceSize([[maybe_unused]] int32_t tokenCount) override { return 0; }
    virtual DimsExprs getOutputDimensions(const DimsExprs *inputs,
                                          [[maybe_unused]] IExprBuilder &exprBuilder) override {
        return nvinfer1::DimsExprs(inputs[0]);
    }
    virtual void copyWeights([[maybe_unused]] void *dst, [[maybe_unused]] int expert,
                             [[maybe_unused]] cudaStream_t stream) override {
        return;
    }
    virtual bool run([[maybe_unused]] int32_t tokenCount, [[maybe_unused]] const void *weights, const void *input,
                     void *output, [[maybe_unused]] void *workspace, cudaStream_t stream) override {
        CUDA_SAFE_CALL(cudaMemcpyAsync(output, input, sizeof(float) * mEmbeddingSize * tokenCount,
                                       cudaMemcpyDeviceToDevice, stream));
        CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
        return true;
    }
    virtual void terminate() { dbg("call terminate"); }
    virtual void initialize() { dbg("call initialize"); }
};

#endif  // IDENTITYLAYER_H
