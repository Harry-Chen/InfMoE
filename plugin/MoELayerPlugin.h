#pragma once

#ifndef MOE_LAYER_PLUGIN_H
#define MOE_LAYER_PLUGIN_H

#include <NvInferPlugin.h>

using namespace nvinfer1;

// plugin specific constants
namespace {
static const char* MOE_LAYER_PLUGIN_VERSION{"1"};
static const char* MOE_LAYER_PLUGIN_NAME{"MoELayerPlugin"};
}  // namespace

class MoELayerPlugin : public IPluginV2 {
   public:
    MoELayerPlugin();
    ~MoELayerPlugin();
    // overloaded virtual functions from IPluginV2
    const char* getPluginType() const TRTNOEXCEPT override { return ::MOE_LAYER_PLUGIN_NAME; };
    const char* getPluginVersion() const TRTNOEXCEPT override { return ::MOE_LAYER_PLUGIN_VERSION; }
    int32_t getNbOutputs() const TRTNOEXCEPT override;
    Dims getOutputDimensions(int32_t index, const Dims* inputs, int32_t nbInputDims) TRTNOEXCEPT override;
    bool supportsFormat(DataType type, PluginFormat format) const TRTNOEXCEPT override;
    void configureWithFormat(const Dims* inputDims, int32_t nbInputs, const Dims* outputDims, int32_t nbOutputs,
                             DataType type, PluginFormat format, int32_t maxBatchSize) TRTNOEXCEPT override;
    int32_t initialize() TRTNOEXCEPT override;
    void terminate() TRTNOEXCEPT override;
    size_t getWorkspaceSize(int32_t maxBatchSize) const TRTNOEXCEPT override;
    int32_t enqueue(int32_t batchSize, const void* const* inputs, void** outputs, void* workspace,
                    cudaStream_t stream) TRTNOEXCEPT override;
    size_t getSerializationSize() const TRTNOEXCEPT override;
    void serialize(void* buffer) const TRTNOEXCEPT override;
    void destroy() TRTNOEXCEPT override;
    IPluginV2* clone() const TRTNOEXCEPT override;
    void setPluginNamespace(const char* pluginNamespace) TRTNOEXCEPT override;
    const char* getPluginNamespace() const TRTNOEXCEPT override;
};

class MoELayerPluginCreator : public IPluginCreator {
   public:
    MoELayerPluginCreator();
    ~MoELayerPluginCreator();
    // overloaded virtual functions from IPluginCreator
    const char* getPluginName() const TRTNOEXCEPT override { return ::MOE_LAYER_PLUGIN_NAME; }
    const char* getPluginVersion() const TRTNOEXCEPT override { return ::MOE_LAYER_PLUGIN_VERSION; }
    const PluginFieldCollection* getFieldNames() TRTNOEXCEPT override;
    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) TRTNOEXCEPT override;
    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) TRTNOEXCEPT override;
    void setPluginNamespace(const char* pluginNamespace) TRTNOEXCEPT override;
    const char* getPluginNamespace() const TRTNOEXCEPT override;
};

#endif  // MOE_LAYER_PLUGIN_H
