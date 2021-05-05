#pragma once

#ifndef T5FFLAYER_H
#define T5FFLAYER_H

#include "../SubLayer.h"

class T5FFLayer : public MoESubLayer {
   private:
    int mEmbeddingSize = -1;
    int mSequenceLength = -1;
   protected:
    virtual ~T5FFLayer() override;

   public:
    virtual bool init(const Dims *inputDims, int32_t nbInputs, const Dims *outputDims, int32_t nbOutputs) override;
    virtual size_t weightSize(int hiddenSize) override;
    virtual size_t workspaceSize(int32_t maxBatchSize) override;
    virtual bool run(int32_t batchSize, const void *weights, const void *const *inputs, void **outputs, void *workspace,
                     cudaStream_t stream, cublasHandle_t cublasHandle) override;
};

#endif  // T5FFLAYER_H