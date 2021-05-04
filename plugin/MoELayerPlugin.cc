#include "MoELayerPlugin.h"

#include <cublas.h>
#include <stdio.h>

#include "utility.h"


void MoELayerPlugin::initializeGPUCentroids() {
    auto size = mExpertCentroidsCPU.count * sizeof(float);
    float* gpu_centroids;
    CUDA_SAFE_CALL(cudaMalloc(&gpu_centroids, size));
    CUDA_SAFE_CALL(cudaMemcpy(gpu_centroids, mExpertCentroidsCPU.values, size, cudaMemcpyHostToDevice));
    mExpertCentroidsGPU = mExpertCentroidsCPU;
    mExpertCentroidsGPU.values = gpu_centroids;
}

MoELayerPlugin::MoELayerPlugin(const char* layerName, int expertCount, int hiddenSize, Weights expertCentroidsCPU,
                               const char* expertWeightFile)
    : mLayerName(strdup(layerName)),
      mExpertCount(expertCount),
      mHiddenSize(hiddenSize),
      mExpertCentroidsCPU(expertCentroidsCPU),
      mExpertWeightFile(expertWeightFile) {
    // check parameters
    assert(mExpertCentroidsCPU.type == DataType::kFLOAT);
    assert(mExpertCentroidsCPU.values != nullptr);
    assert(mExpertCentroidsCPU.count > 0);
}

MoELayerPlugin::MoELayerPlugin(const MoELayerPlugin& src)
    : MoELayerPlugin(strdup(src.mLayerName), mExpertCount, mHiddenSize, mExpertCentroidsCPU, strdup(src.mExpertWeightFile)) {
    // copy centroids
    auto size = mExpertCentroidsCPU.count * sizeof(float);
    float* cpu_centroids = static_cast<float*>(malloc(size));
    memcpy(cpu_centroids, src.mExpertCentroidsCPU.values, size);
    mExpertCentroidsCPU.values = cpu_centroids;
    initializeGPUCentroids();
}

MoELayerPlugin::MoELayerPlugin(const char* layerName, const void* serialData, size_t serialLength): mLayerName(strdup(layerName)) {
    assert(serialLength >= METADATA_LENGTH);
    auto int_buffer = reinterpret_cast<const int *>(serialData);
    mExpertCount = *int_buffer++;
    mHiddenSize = *int_buffer;
    auto int64_buffer = reinterpret_cast<const int64_t *>(int_buffer);
    // initialize centroids
    mExpertCentroidsCPU.count = *int64_buffer++;
    auto size = mExpertCentroidsCPU.count * sizeof(float);\
    assert(size == serialLength - METADATA_LENGTH);
    float* cpu_centroids = static_cast<float*>(malloc(size));
    memcpy(cpu_centroids, int64_buffer, mExpertCentroidsCPU.count * sizeof(float));
    mExpertCentroidsCPU.values = cpu_centroids;
    initializeGPUCentroids();
}

MoELayerPlugin::~MoELayerPlugin() { terminate(); }

Dims MoELayerPlugin::getOutputDimensions(int32_t index, const Dims* inputs, int32_t nbInputDims) noexcept {
    assert(index == 0);
    assert(nbInputDims == 0);
    // output tensor should have the same shape with input tensor
    return inputs[0];
}

bool MoELayerPlugin::supportsFormat(DataType type, PluginFormat format) const noexcept {
    return type == DataType::kFLOAT && format == PluginFormat::kLINEAR;
}

void MoELayerPlugin::configureWithFormat(const Dims* inputDims, int32_t nbInputs, const Dims* outputDims,
                                         int32_t nbOutputs, DataType type, PluginFormat format,
                                         int32_t maxBatchSize) noexcept {
    assert(nbInputs == 1 && nbOutputs == 1 && type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
    // outputDims[0] should equal inputDims[0]
    assert(outputDims[0].nbDims == inputDims[0].nbDims);
    auto& dim = inputDims[0];
    // check and set dimensions: should be token * embedding
    assert(dim.nbDims == 2);
    mEmbeddingSize = dim.d[0];
    mSequenceLength = dim.d[1];
    mMaxBatchSize = maxBatchSize;
}

int32_t MoELayerPlugin::initialize() noexcept {
    // initialize cublas
    CUBLAS_SAFE_CALL(cublasCreate_v2(&mCublasHandle));
    assert(mCublasHandle != nullptr);
    return 0;
}

void MoELayerPlugin::terminate() noexcept {
    // free centroids on CPU and GPU
    free(const_cast<void*>(mExpertCentroidsCPU.values));
    CUDA_SAFE_CALL(cudaFree(const_cast<void*>(mExpertCentroidsGPU.values)));
    mExpertCentroidsGPU.values = nullptr;
    mExpertCentroidsCPU.values = nullptr;
    // destroy cublas handle
    CUBLAS_SAFE_CALL(cublasDestroy_v2(mCublasHandle));
    mCublasHandle = nullptr;
}

size_t MoELayerPlugin::getWorkspaceSize(int32_t maxBatchSize) const noexcept {
    // TODO: calculate workspace size needed
    unimplemented();
}

int32_t MoELayerPlugin::enqueue(int32_t batchSize, const void* const* inputs, void** outputs, void* workspace,
                                cudaStream_t stream) noexcept {
    // run the actual MoE calculation
    // 1. calculate token-expert routing
    // 2. get top-1 choice
    // 3. sort & gather (a.k.a. shuffle) tokens for each export
    // 4. run each expert
    //    state = state + dense_relu_dense(layer_norm(state))
    //    dense_relu_dense: hs = wo * (gelu(wi_0 * hs) * (wi_1 * hs))
    // 5. unshuffle results
    unimplemented();
}

size_t MoELayerPlugin::getSerializationSize() const noexcept {
    return METADATA_LENGTH + mExpertCentroidsCPU.count * sizeof(float);
}

void MoELayerPlugin::serialize(void* buffer) const noexcept {
    auto int_buffer = reinterpret_cast<int *>(buffer);
    *int_buffer++ = mExpertCount;
    *int_buffer++ = mHiddenSize;
    auto int64_buffer = reinterpret_cast<int64_t *>(int_buffer);
    *int64_buffer++ = mExpertCentroidsCPU.count;
    memcpy(int64_buffer, mExpertCentroidsCPU.values, mExpertCentroidsCPU.count * sizeof(float));
}

void MoELayerPlugin::destroy() noexcept { delete this; }

IPluginV2* MoELayerPlugin::clone() const noexcept { return new MoELayerPlugin(*this); }

void MoELayerPlugin::setPluginNamespace(const char* pluginNamespace) noexcept { mPluginNamespace = pluginNamespace; }

const char* MoELayerPlugin::getPluginNamespace() const noexcept { return mPluginNamespace; }
