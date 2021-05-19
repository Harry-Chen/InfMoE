#pragma once

#ifndef T5FFLAYER_H
#define T5FFLAYER_H

#include "../thirdparty/cnpy/cnpy.h"

#include "../SubLayer.h"

class T5FFLayer : public MoESubLayer {
   private:
    int mEmbeddingSize;
    int mSequenceLength;
   
   // weights
   private:
    cnpy::npz_t *mSavedWeights;

   public:
    using MoESubLayer::MoESubLayer;
    virtual ~T5FFLayer() override;
    virtual bool init(const Dims *inputDims, int32_t nbInputs, const Dims *outputDims, int32_t nbOutputs) override;
    virtual size_t weightSize() override;
    virtual size_t workspaceSize(int32_t maxBatchSize) override;
    virtual Dims getOutputDimensions(int32_t index, const Dims* inputs, int32_t nbInputDims) override;
    void copyWeights(void *dst, int expert, cudaStream_t stream) override;
    virtual bool run(int32_t batchSize, const void *weights, const void *const *inputs, void **outputs, void *workspace,
                     cudaStream_t stream) override;
    void initialize();
    void terminate();
};

#endif  // T5FFLAYER_H
