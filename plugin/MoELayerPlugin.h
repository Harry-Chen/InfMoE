#pragma once

#ifndef MOE_LAYER_PLUGIN_H
#define MOE_LAYER_PLUGIN_H

#include <array>
#include <NvInferPlugin.h>
#include <cublas.h>

#include "SubLayer.h"

using namespace nvinfer1;

// plugin specific constants
namespace {
static const char* MOE_LAYER_PLUGIN_VERSION{"1"};
static const char* MOE_LAYER_PLUGIN_NAME{"MoELayerPlugin"};
}  // namespace


namespace sublayer_type {
static const char *T5FF{"T5_FF"};
}

class MoELayerPlugin : public IPluginV2 {

   private:
    // TensorRT / CUDA related
    const char *mLayerName = nullptr;
    const char *mPluginNamespace = nullptr;
    int mMaxBatchSize = -1;
    cublasHandle_t mCublasHandle = nullptr;
    cudaStream_t *mStreams;

    // layer parameters
    int mExpertCount;
    int mHiddenSize;
    int mMaxConcurrency = 2; // maximum number of sublayers on GPU memory
    Weights mExpertCentroidsCPU, mExpertCentroidsGPU;
    const char *mExpertWeightFile, *mSublayerType;
    MoESubLayer *mSublayer;
    mutable size_t mSublayerWorkspacecSize;
    
    // inferred from network
    int mEmbeddingSize = -1;
    int mSequenceLength = -1;
    void initializeGPUCentroids();
    constexpr const static size_t METADATA_LENGTH = sizeof(mExpertCount) + sizeof(mHiddenSize) + sizeof(mExpertCentroidsCPU.count) + sizeof(int) * 2;

   public:
    // constructor for MoELayerPluginCreator
    explicit MoELayerPlugin(const char *layerName, int expertCount, int hiddenSize, Weights expertCentroidsCPU, const char *expertWeightFile, const char *sublayerType);
    // constructor for clone
    explicit MoELayerPlugin(const MoELayerPlugin &src);
    // constructor for deserialization
    explicit MoELayerPlugin(const char *layerName, const void* serialData, size_t serialLength);
    // destructor
    ~MoELayerPlugin();
    // overloaded virtual functions from IPluginV2
    const char* getPluginType() const noexcept override { return ::MOE_LAYER_PLUGIN_NAME; };
    const char* getPluginVersion() const noexcept override { return ::MOE_LAYER_PLUGIN_VERSION; }
    int32_t getNbOutputs() const noexcept override { return 1; }
    // implemented in .cc file
    Dims getOutputDimensions(int32_t index, const Dims* inputs, int32_t nbInputDims) noexcept override;
    bool supportsFormat(DataType type, PluginFormat format) const noexcept override;
    void configureWithFormat(const Dims* inputDims, int32_t nbInputs, const Dims* outputDims, int32_t nbOutputs,
                             DataType type, PluginFormat format, int32_t maxBatchSize) noexcept override;
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    size_t getWorkspaceSize(int32_t maxBatchSize) const noexcept override;
    int32_t enqueue(int32_t batchSize, const void* const* inputs, void** outputs, void* workspace,
                    cudaStream_t stream) noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    IPluginV2* clone() const noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;
};

class MoELayerPluginCreator : public IPluginCreator {
   private:
    const char *mPluginNamespace = nullptr;
    const static std::array<PluginField, 5> mPluginAttributes;
    const static PluginFieldCollection mFC;
   public:
    MoELayerPluginCreator();
    ~MoELayerPluginCreator();
    // overloaded virtual functions from IPluginCreator
    const char* getPluginName() const noexcept override { return ::MOE_LAYER_PLUGIN_NAME; }
    const char* getPluginVersion() const noexcept override { return ::MOE_LAYER_PLUGIN_VERSION; }
    // implemented in .cc file
    const PluginFieldCollection* getFieldNames() noexcept override;
    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;
    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;
};

#endif  // MOE_LAYER_PLUGIN_H
