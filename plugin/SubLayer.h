#pragma once

#ifndef SUBLAYER_H
#define SUBLAYER_H

#include <cublas.h>
#include <NvInferPlugin.h>

using nvinfer1::Dims;
using nvinfer1::Weights;

class MoESubLayer {
   protected:
    virtual ~MoESubLayer();

   public:
    virtual bool init(const Dims *inputDims, int32_t nbInputs, const Dims *outputDims, int32_t nbOutputs) = 0;
    virtual size_t weightSize(int hiddenSize) = 0;
    virtual size_t workspaceSize(int32_t maxBatchSize) = 0;
    virtual bool run(int32_t batchSize, const void *weights, const void *const *inputs, void **outputs, void *workspace,
                     cudaStream_t stream, cublasHandle_t cublasHandle) = 0;
};

#endif  // SUBLAYER_H
