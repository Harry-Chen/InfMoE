#pragma once

#ifndef IDENTITYLAYER_H
#define IDENTITYLAYER_H

#include <cuda_runtime.h>
#include <NvInferPlugin.h>

#include "SubLayer.h"
#include "../thirdparty/dbg.h"

class IdentityLayer : public MoESubLayer {
   private:
    int mEmbeddingSize;

   public:
    explicit IdentityLayer() : MoESubLayer(0, 0, nullptr, 0) {}
    virtual ~IdentityLayer() override {}
    virtual bool configureWithFormat(const Dims *inputDims, int32_t nbInputs, const Dims *outputDims,
                                     int32_t nbOutputs) override {
        assert(outputDims[0].nbDims == inputDims[0].nbDims && inputDims[0].nbDims == 3);
        auto &dim = inputDims[0];
        mEmbeddingSize = dim.d[2];
        return true;
    }
    virtual size_t weightSize() override { return 0; }
    virtual size_t workspaceSize(int32_t tokenCount) override { return 0; }
    virtual DimsExprs getOutputDimensions(const DimsExprs* inputs, IExprBuilder& exprBuilder) override {
        return nvinfer1::DimsExprs(inputs[0]);
    }
    virtual void copyWeights(void *dst, int expert, cudaStream_t stream) override { return; }
    virtual bool run(int32_t tokenCount, const void *weights, const void *input, void *output, void *workspace,
                     cudaStream_t stream) override {
        CUDA_SAFE_CALL(cudaMemcpyAsync(output, input, sizeof(float) * mEmbeddingSize * tokenCount,
                                       cudaMemcpyDeviceToDevice, stream));
        CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
    }
    virtual void terminate() { dbg("call terminate"); }
    virtual void initialize() { dbg("call initialize"); }
};

#endif  // IDENTITYLAYER_H
