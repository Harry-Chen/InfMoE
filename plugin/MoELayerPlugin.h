#pragma once

#ifndef MOE_LAYER_PLUGIN_H
#define MOE_LAYER_PLUGIN_H

#include <NvInferPlugin.h>
#include <cublas_v2.h>

#include <memory>
#include <array>

#include "sublayers/SubLayer.h"

using namespace nvinfer1;

// plugin specific constants
namespace {
static const char* MOE_LAYER_PLUGIN_VERSION{"1"};
static const char* MOE_LAYER_PLUGIN_NAME{"MoELayerPlugin"};
}  // namespace


namespace sublayer_type {
[[maybe_unused]] static const char* T5FF{"T5_FF"};
[[maybe_unused]] static const char* Identity{"Identity"};
}  // namespace sublayer_type

namespace moe_variant {
[[maybe_unused]] static const char* BASE_LAYER{"base_layer"}; // no preprocess on input, mix expert-output with input by sigmoid(score)
[[maybe_unused]] static const char* CPM_2{"cpm_2"}; // score = layernorm(input) @ centroid, no mix
[[maybe_unused]] static const char* DEFAULT{"default"}; // no preprocess on input, no mix
} // namespace moe_variant


// store behaviour flags of MoE layers
struct MoEFlags {
    bool layernormOnInputBeforeScore = false;
    bool baseLayerOutputMix= false;
    uint16_t padding = 0;
};

static_assert(sizeof(MoEFlags) == 4);


class MoELayerPlugin : public IPluginV2DynamicExt  {

   private:
    // TensorRT / CUDA related
    const char* mLayerName = nullptr;
    const char* mPluginNamespace = nullptr;
    cublasHandle_t mCublasHandle = nullptr;
    cudaStream_t* mStreams = nullptr;

    // layer parameters
    int mExpertCount;
    int mHiddenSize;
    int mMaxConcurrency;  // maximum number of sublayers on GPU memory
    Weights mExpertCentroidsCPU{}, mExpertCentroidsGPU{};
    const char *mExpertWeightFile, *mSublayerType;
    MoEFlags mFlags; // store other flags

    // sublayer related
    std::shared_ptr<MoESubLayer> mSublayer = nullptr;
    mutable size_t mSublayerWorkspacecSize;

    // inferred from network
    int mEmbeddingSize = -1;
    int mSequenceLength = -1;
    void ensureGPUCentroids();
    void ensureSublayerWorkspaceSize(size_t tokenCount) const;
    void createSublayer();
    void ensureCUDAContext();
    constexpr const static size_t METADATA_LENGTH = sizeof(mExpertCount) + sizeof(mHiddenSize) + sizeof(mFlags) +
                                                    sizeof(mMaxConcurrency) + sizeof(mExpertCentroidsCPU.count) +
                                                    sizeof(int) * 2;

   public:
    // constructor for MoELayerPluginCreator
    explicit MoELayerPlugin(const char* layerName, int expertCount, int hiddenSize, int maxConcurrency,
                            Weights expertCentroidsCPU, const char* expertWeightFile, const char* sublayerType, const MoEFlags flags);
    // constructor for clone
    explicit MoELayerPlugin(const MoELayerPlugin& src);
    // constructor for deserialization
    explicit MoELayerPlugin(const char* layerName, const void* serialData, size_t serialLength);
    // destructor
    virtual ~MoELayerPlugin();
    // parse flags from variant
    static MoEFlags parseFlags(const char* moeVariant);
    // overloaded virtual functions from IPluginV2
    const char* getPluginType() const noexcept override { return ::MOE_LAYER_PLUGIN_NAME; };
    const char* getPluginVersion() const noexcept override { return ::MOE_LAYER_PLUGIN_VERSION; }
    int32_t getNbOutputs() const noexcept override { return 1; }
    // implemented in .cc file
    // IPluginV2
    // Dims getOutputDimensions(int32_t index, const Dims* inputs, int32_t nbInputDims) noexcept override;
    // bool supportsFormat(DataType type, PluginFormat format) const noexcept override;
    // void configureWithFormat(const Dims* inputDims, int32_t nbInputs, const Dims* outputDims, int32_t nbOutputs,
    //                          DataType type, PluginFormat format, int32_t maxBatchSize) noexcept override;
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    // size_t getWorkspaceSize(int32_t maxBatchSize) const noexcept override;
    // int32_t enqueue(int32_t batchSize, void const* const* inputs, void* const* outputs, void* workspace,
    //                 cudaStream_t stream) noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    IPluginV2DynamicExt* clone() const noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;
    // IPluginV2Ext
    nvinfer1::DataType getOutputDataType(int32_t index, nvinfer1::DataType const* inputTypes,
                                         int32_t nbInputs) const noexcept override;
    // void attachToContext(cudnnContext* /*cudnn*/, cublasContext* /*cublas*/, IGpuAllocator* /*allocator*/) noexcept override;
    // void detachFromContext() noexcept override;
    // bool isOutputBroadcastAcrossBatch(int32_t outputIndex, bool const* inputIsBroadcasted,
    //                                   int32_t nbInputs) const noexcept override;
    // bool canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept override;
    // void configurePlugin(Dims const* inputDims, int32_t nbInputs, Dims const* outputDims, int32_t nbOutputs,
    //                      DataType const* inputTypes, DataType const* outputTypes, bool const* inputIsBroadcast,
    //                      bool const* outputIsBroadcast, PluginFormat floatFormat,
    //                      int32_t maxBatchSize) noexcept override;
    // IPluginV2DynamicExt
    DimsExprs getOutputDimensions(
        int32_t outputIndex, const DimsExprs* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    void configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInputs,
        const DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(const PluginTensorDesc* inputs, int32_t nbInputs, const PluginTensorDesc* outputs,
        int32_t nbOutputs) const noexcept override;
    int32_t enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
};

class MoELayerPluginCreator : public IPluginCreator {
   private:
    const char* mPluginNamespace = nullptr;
    const static std::array<PluginField, 7> mPluginAttributes;
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
