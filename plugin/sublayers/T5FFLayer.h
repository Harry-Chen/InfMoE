#pragma once

#ifndef T5FFLAYER_H
#define T5FFLAYER_H

#include <cuda_runtime.h>

#include "../thirdparty/cnpy/cnpy.h"
#include "../SubLayer.h"

class T5FFLayer : public MoESubLayer {
   private:
    int mEmbeddingSize;
    int mSequenceLength;
    cudaDeviceProp mDeviceProp;
   
   // weights
   private:
    cnpy::npz_t *mSavedWeights = nullptr;
    size_t layernormWeightSize() const {
        return mEmbeddingSize * sizeof(float);
    }
    size_t intermediateFFWeightSize() const {
        return mEmbeddingSize * mHiddenSize * sizeof(float);
    }
    size_t layernormOutputSize(int32_t tokenCount) const {
        return tokenCount * mEmbeddingSize * sizeof(float);
    }
    size_t intermediateFFOutputSize(int32_t tokenCount) const {
        return tokenCount * mHiddenSize * sizeof(float);
    }

   public:
    using MoESubLayer::MoESubLayer;
    virtual ~T5FFLayer() override;
    virtual bool configureWithFormat(const Dims *inputDims, int32_t nbInputs, const Dims *outputDims, int32_t nbOutputs) override;
    virtual size_t weightSize() override;
    virtual size_t workspaceSize(int32_t tokenCount) override;
    virtual Dims getOutputDimensions(int32_t index, const Dims* inputs, int32_t nbInputDims) override;
    virtual void copyWeights(void *dst, int expert, cudaStream_t stream) override;
    virtual bool run(int32_t tokenCount, const void *weights, const void *input, void *output, void *workspace,
                     cudaStream_t stream) override;
    void initialize();
    void terminate();
};

#endif  // T5FFLAYER_H
