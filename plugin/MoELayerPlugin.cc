#include "MoELayerPlugin.h"

#include <cublas.h>
#include <stdio.h>

#include "utility.h"
#include "sublayers/T5FFLayer.h"

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
    float* cpu_centroids = new float[size];
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
    float* cpu_centroids = new float[size];
    memcpy(cpu_centroids, int64_buffer, mExpertCentroidsCPU.count * sizeof(float));
    mExpertCentroidsCPU.values = cpu_centroids;
    initializeGPUCentroids();
    // initialize sublayer
    // TODO: read specific layer from config
    mSublayer = new T5FFLayer(mExpertCount, mHiddenSize, mExpertWeightFile, mMaxConcurrency, &mCublasHandle);
}

MoELayerPlugin::~MoELayerPlugin() { terminate(); }

Dims MoELayerPlugin::getOutputDimensions(int32_t index, const Dims* inputs, int32_t nbInputDims) noexcept {
    return mSublayer->getOutputDimensions(index, inputs, nbInputDims);
}

bool MoELayerPlugin::supportsFormat(DataType type, PluginFormat format) const noexcept {
    return type == DataType::kFLOAT && format == PluginFormat::kLINEAR;
}

void MoELayerPlugin::configureWithFormat(const Dims* inputDims, int32_t nbInputs, const Dims* outputDims,
                                         int32_t nbOutputs, DataType type, PluginFormat format,
                                         int32_t maxBatchSize) noexcept {
    assert(nbInputs == 1 && nbOutputs == 1 && type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
    assert(mSublayer->init(inputDims, nbInputs, outputDims, nbOutputs));
    auto &dim = inputDims[0];
    auto &dim2 = outputDims[0];
    mEmbeddingSize = dim.d[1];
    mSequenceLength = dim.d[0];
    mMaxBatchSize = maxBatchSize;
}

int32_t MoELayerPlugin::initialize() noexcept {
    CUBLAS_SAFE_CALL(cublasCreate_v2(&mCublasHandle));
    mStreams = new cudaStream_t[mMaxConcurrency];
    for (int i = 0; i < mMaxConcurrency; ++i) {
        CUDA_SAFE_CALL(cudaStreamCreate(&mStreams[i]));
    }
    assert(mCublasHandle != nullptr);
    mSublayer->initialize();
    return 0;
}

void MoELayerPlugin::terminate() noexcept {
    // free centroids on CPU and GPU
    delete[] mExpertCentroidsCPU.values;
    CUDA_SAFE_CALL(cudaFree(const_cast<void*>(mExpertCentroidsGPU.values)));
    mExpertCentroidsGPU.values = nullptr;
    mExpertCentroidsCPU.values = nullptr;
    // destroy cublas handle
    CUBLAS_SAFE_CALL(cublasDestroy_v2(mCublasHandle));
    mCublasHandle = nullptr;
    // destroy cuda streams
    for (int i = 0; i < mMaxConcurrency; ++i) {
        CUDA_SAFE_CALL(cudaStreamDestroy(mStreams[i]));
    }
    delete[] mStreams;
    // invoke sublayer termination
    mSublayer->terminate();
}

size_t MoELayerPlugin::getWorkspaceSize(int32_t maxBatchSize) const noexcept {
    mSublayerWorkspacecSize = mSublayer->workspaceSize(maxBatchSize) + mSublayer->weightSize();
    auto sublayer_size = mSublayerWorkspacecSize * mMaxConcurrency;
    auto plugin_size = 0; // TODO: confirm
    return plugin_size + sublayer_size;
}

int32_t MoELayerPlugin::enqueue(int32_t batchSize, const void* const* inputs, void** outputs, void* workspace,
                                cudaStream_t stream) noexcept {
    // TODO: run the actual MoE calculation
    // 1. calculate token-expert routing
    // 2. get top-1 choice
    // 3. sort & gather (a.k.a. shuffle) tokens for each expert

    // 4. run each expert: state = sublayer.run(state)
    mSublayer->copyWeights(workspace, 0, mStreams[0]);

    for (int i = 0; i < mExpertCount; ++i) {
        auto workspace_byte = reinterpret_cast<char*>(workspace);
        auto current_idx = i % mMaxConcurrency;
        auto next_idx = (i + 1) % mMaxConcurrency;
        auto current_workspace = workspace_byte + mSublayerWorkspacecSize * current_idx;
        auto next_workspace = workspace_byte + mSublayerWorkspacecSize * next_idx;
        CUDA_SAFE_CALL(cudaStreamSynchronize(mStreams[next_idx]));
        if (i != mExpertCount - 1) {
            mSublayer->copyWeights(next_workspace, i + 1, mStreams[next_idx]);
        }
        // TODO: calcualte input and output address
        assert(mSublayer->run(batchSize, current_workspace, nullptr, nullptr, current_workspace + mSublayer->weightSize(), mStreams[current_idx]));
    }

    // synchronize all streams
    for (int i = 0; i < mMaxConcurrency; ++i) {
        CUDA_SAFE_CALL(cudaStreamSynchronize(mStreams[i]));
    }

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
