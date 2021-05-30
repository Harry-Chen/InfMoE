#pragma once

#ifndef SUBLAYER_H
#define SUBLAYER_H

#include <NvInferPlugin.h>
#include <cublas_v2.h>

#include <cassert>

#include "utility.h"

using nvinfer1::Dims;
using nvinfer1::Weights;
using nvinfer1::DimsExprs;
using nvinfer1::IExprBuilder;

class MoESubLayer {
   protected:
    int mExpertCount;
    int mEmbeddingSize;
    int mHiddenSize;
    int mMaxConcurrency;
    const char *mWeightFile;
    cublasHandle_t mCublasHandle = nullptr;  // passed by MoELayerPlugin

   public:
    explicit MoESubLayer(int expertCount, int embeddingSize, int hiddenSize, const char *weightFile, int maxConcurrency)
        : mExpertCount(expertCount),
          mEmbeddingSize(embeddingSize),
          mHiddenSize(hiddenSize),
          mMaxConcurrency(maxConcurrency),
          mWeightFile(weightFile) {};
    void setCuBlasHandle(cublasHandle_t handle) { mCublasHandle = handle; }
    virtual ~MoESubLayer(){};
    virtual bool configureWithFormat(const Dims *inputDims, int32_t nbInputs, const Dims *outputDims, int32_t nbOutputs) = 0;
    virtual size_t weightSize() = 0;
    virtual size_t workspaceSize(int32_t tokenCount) = 0;
    virtual DimsExprs getOutputDimensions(const DimsExprs* inputs, IExprBuilder& exprBuilder) = 0;
    virtual void copyWeights(void *dst, int expert, cudaStream_t stream) = 0;
    virtual bool run(int32_t tokenCount, const void *weights, const void *input, void *output, void *workspace,
                     cudaStream_t stream) = 0;
    // read weights to memory, etc.
    virtual void initialize() = 0;
    // free weights, etc.
    virtual void terminate() = 0;
};

#endif  // SUBLAYER_H
