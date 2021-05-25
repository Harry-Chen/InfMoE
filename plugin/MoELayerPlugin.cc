#include "MoELayerPlugin.h"

#include <cublas_v2.h>
#include <stdio.h>

#include "thirdparty/dbg.h"
#include "cuda/moe.h"
#include "sublayers/T5FFLayer.h"
#include "utility.h"

void MoELayerPlugin::initializeGPUCentroids() {
    auto size = mExpertCentroidsCPU.count * sizeof(float);
    float* gpu_centroids;
    CUDA_SAFE_CALL(cudaMalloc(&gpu_centroids, size));
    CUDA_SAFE_CALL(cudaMemcpy(gpu_centroids, mExpertCentroidsCPU.values, size, cudaMemcpyHostToDevice));
    mExpertCentroidsGPU = mExpertCentroidsCPU;
    mExpertCentroidsGPU.values = gpu_centroids;
}

void MoELayerPlugin::createSublayer() {
    assert(mSublayerType != nullptr);
    assert(mSublayer == nullptr);
    // initialize sublayer according to parameter
    if (strcmp(mSublayerType, sublayer_type::T5FF) == 0) {
        mSublayer = new T5FFLayer(mExpertCount, mHiddenSize, mExpertWeightFile, mMaxConcurrency, &mCublasHandle);
    } else {
        fprintf(stderr, "unsupported sublayer type: %s\n", mSublayerType);
        assert(false);
    }
}

MoELayerPlugin::MoELayerPlugin(const char* layerName, int expertCount, int hiddenSize, int maxConcurrency,
                               Weights expertCentroidsCPU, const char* expertWeightFile, const char *sublayerType)
    : mLayerName(strdup(layerName)),
      mExpertCount(expertCount),
      mHiddenSize(hiddenSize),
      mMaxConcurrency(maxConcurrency),
      mExpertCentroidsCPU(expertCentroidsCPU),
      mExpertWeightFile(expertWeightFile),
      mSublayerType(strdup(sublayerType)) {
    dbg(this, "construct MoELayerPlugin from creator");
    // check parameters
    assert(mExpertCentroidsCPU.type == DataType::kFLOAT);
    assert(mExpertCentroidsCPU.values != nullptr);
    assert(mExpertCentroidsCPU.count > 0);
    initializeGPUCentroids();
    createSublayer();
}

MoELayerPlugin::MoELayerPlugin(const MoELayerPlugin& src)
    : MoELayerPlugin(strdup(src.mLayerName), src.mExpertCount, src.mHiddenSize, src.mMaxConcurrency, src.mExpertCentroidsCPU,
                     strdup(src.mExpertWeightFile), strdup(src.mSublayerType)) {
    dbg(this, "construct MoELayerPlugin from copy constructor");
    // WORKAROUND
    mSublayer = nullptr;
    // copy centroids
    auto size = src.mExpertCentroidsCPU.count * sizeof(float);
    float* cpu_centroids = new float[size];
    memcpy(cpu_centroids, src.mExpertCentroidsCPU.values, size);
    mExpertCentroidsCPU.values = cpu_centroids;
    initializeGPUCentroids();
    createSublayer();
}

MoELayerPlugin::MoELayerPlugin(const char* layerName, const void* serialData, size_t serialLength)
    : mLayerName(strdup(layerName)) {
    dbg(this, "construct MoELayerPlugin from serialized data");
    assert(serialLength >= METADATA_LENGTH);
    // 5 int32_t
    auto int_buffer = reinterpret_cast<const int*>(serialData);
    mExpertCount = *int_buffer++;
    mHiddenSize = *int_buffer++;
    mMaxConcurrency = *int_buffer++;
    auto expert_weight_file_len = *int_buffer++;
    auto sublayer_type_len = *int_buffer++;
    // 2 strings
    auto char_buffer = reinterpret_cast<const char*>(int_buffer);
    mExpertWeightFile = strdup(char_buffer);
    char_buffer += expert_weight_file_len + 1;
    mSublayerType = strdup(char_buffer);
    char_buffer += sublayer_type_len + 1;
    auto string_size = expert_weight_file_len + sublayer_type_len + 2;
    // align to 8 byte
    if (string_size % 8 != 0) {
        char_buffer += 8 - (string_size % 8);
    }
    // initialize centroids (int64_t)
    auto int64_buffer = reinterpret_cast<const int64_t*>(char_buffer);
    mExpertCentroidsCPU.count = *int64_buffer++;
    auto size = mExpertCentroidsCPU.count * sizeof(float);
    assert(size == serialLength - METADATA_LENGTH);
    float* cpu_centroids = new float[size];
    memcpy(cpu_centroids, int64_buffer, mExpertCentroidsCPU.count * sizeof(float));
    mExpertCentroidsCPU.values = cpu_centroids;
    initializeGPUCentroids();
    createSublayer();
}

MoELayerPlugin::~MoELayerPlugin() { 
    dbg(this, "destructing MoELayerPlugin");
    terminate(); 
}

Dims MoELayerPlugin::getOutputDimensions(int32_t index, const Dims* inputs, int32_t nbInputDims) noexcept {
    dbg(index, nbInputDims);
    return mSublayer->getOutputDimensions(index, inputs, nbInputDims);
}

bool MoELayerPlugin::supportsFormat(DataType type, PluginFormat format) const noexcept {
    dbg(type, format);
    return type == DataType::kFLOAT && format == PluginFormat::kLINEAR;
}

void MoELayerPlugin::configureWithFormat(const Dims* inputDims, int32_t nbInputs, const Dims* outputDims,
                                         int32_t nbOutputs, DataType type, PluginFormat format,
                                         int32_t maxBatchSize) noexcept {
    dbg(maxBatchSize);
    assert(nbInputs == 1 && nbOutputs == 1 && type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
    assert(mSublayer->configureWithFormat(inputDims, nbInputs, outputDims, nbOutputs));
    auto& dim = inputDims[0];
    mEmbeddingSize = dim.d[1];
    mSequenceLength = dim.d[0];
    mMaxBatchSize = maxBatchSize;
}

int32_t MoELayerPlugin::initialize() noexcept {
    dbg(this, "call initialize");
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
    dbg(this, "call terminate");
    // free centroids on CPU and GPU
    if (mExpertCentroidsCPU.values != nullptr) {
        delete[] static_cast<float*>(const_cast<void*>(mExpertCentroidsCPU.values));
        mExpertCentroidsGPU.values = nullptr;
    }
    if (mExpertCentroidsGPU.values != nullptr) {
        CUDA_SAFE_CALL(cudaFree(const_cast<void*>(mExpertCentroidsGPU.values)));
        mExpertCentroidsCPU.values = nullptr;
    }
    // destroy cublas handle
    if (mCublasHandle != nullptr) {
        CUBLAS_SAFE_CALL(cublasDestroy_v2(mCublasHandle));
        mCublasHandle = nullptr;
    }
    // destroy cuda streams
    if (mStreams != nullptr) {
        for (int i = 0; i < mMaxConcurrency; ++i) {
            CUDA_SAFE_CALL(cudaStreamDestroy(mStreams[i]));
        }
        delete[] mStreams;
    }
    // invoke sublayer termination
    if (mSublayer != nullptr) {
        mSublayer->terminate();
        delete mSublayer;
        mSublayer = nullptr;
    }
}

// GPU workspace is consists of:
// 1. maxConcurrency times of layer workspace (weights + intermedaite variables)
// 2. MoE buffer, including:
//     a. token-gate affiliation (token_num * expert_count) where token_num = batch_size * seq_len
//     b. gate selection (int, token_num)
//     c. token original position (int, token_num)
//     d. routed features (token_num * d_model)
//     e. routed features after expert (token_num * d_model)
//     f. 2 * coefficient to mix routed features after & before expert (token_num)
// They will not be simultaneously used, so we take the max of two space
size_t MoELayerPlugin::getWorkspaceSize(int32_t maxBatchSize) const noexcept {
    // the maximum tokens that might go to one single expert
    // FIXME: hardcoded (as claimed in BASE Layer paper)
    auto max_single_expert_token_count = static_cast<int32_t>(maxBatchSize * mSequenceLength / mExpertCount * 5);
    mSublayerWorkspacecSize = mSublayer->weightSize() + mSublayer->workspaceSize(max_single_expert_token_count);
    auto sublayer_size = mSublayerWorkspacecSize * mMaxConcurrency;
    // maximum tokens that might be processed by this layer
    auto max_token_count = maxBatchSize * mSequenceLength;
    auto plugin_size =
        (max_token_count * mExpertCount + max_token_count * 2 + max_token_count * mEmbeddingSize * 2) * sizeof(float) +
        max_token_count * 2 * sizeof(int);
    auto final_size = std::max(plugin_size, sublayer_size);
    dbg(final_size);
    return final_size;
}

int32_t MoELayerPlugin::enqueue(int32_t batchSize, void const* const* inputs, void* const* outputs, void* workspace,
                                cudaStream_t stream) noexcept {
    dbg("call enqueue");
    // run the actual MoE calculation
    // 0. obtain all buffers
    auto token_num = batchSize * mSequenceLength;
    auto token_len = mEmbeddingSize;
    auto d_layer_input = static_cast<const float*>(inputs[0]);
    auto d_expert_centroids = static_cast<const float*>(mExpertCentroidsGPU.values);
    auto moe_buffer = reinterpret_cast<float*>(static_cast<char*>(workspace));
    auto d_token_expert_aff = moe_buffer;
    auto d_gate_selection = reinterpret_cast<int*>(moe_buffer + token_num * mExpertCount);
    auto d_token_pos = d_gate_selection + token_num;
    auto d_routed_features = reinterpret_cast<float*>(d_token_pos + token_num);
    auto d_post_expert_features = d_routed_features + token_num * token_len;
    auto d_mix_coeff = d_post_expert_features + token_num * token_len;
    auto d_routed_mix_coeff = d_mix_coeff + token_num;
    auto d_layer_output = static_cast<float*>(outputs[0]);

    // 1. calculate token-expert affiliation
    // (token_num, token_len) @ (token_len, expert_count)
    float alpha = 1.0, beta = 0.0;
    CUBLAS_SAFE_CALL(cublasSetStream_v2(mCublasHandle, stream));
    CUBLAS_SAFE_CALL(cublasSgemm_v2(mCublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, mExpertCount, token_num, token_len, &alpha,
                                    d_expert_centroids, token_len, d_layer_input, token_len, &beta, d_token_expert_aff,
                                    mExpertCount));
    CUDA_SAFE_CALL(cudaStreamSynchronize(stream));

    dbg("after centroids");
    // 2. get expert assignments (TODO: support multiple experts for each token)
    moe_expert_select(token_num, token_len, d_token_expert_aff, d_gate_selection, d_mix_coeff, stream);
    dbg("after select");

    // 3. count & sort & gather (a.k.a. shuffle) tokens for each expert
    auto expert_offset = new int[mExpertCount + 1]();
    expert_offset[mExpertCount] = token_num;
    moe_expert_count(token_num, mExpertCount, d_gate_selection, d_token_pos, expert_offset, stream);
    dbg("after count");
    moe_expert_scatter(token_num, token_len, d_layer_input, d_mix_coeff, d_token_pos, d_routed_features,
                       d_routed_mix_coeff, stream);
    dbg("after scatter");

    // 4. run each expert: state = sublayer.run(state)
    mSublayer->copyWeights(workspace, 0, mStreams[0]);

    for (int i = 0; i < mExpertCount; ++i) {
        auto current_token_offset = expert_offset[i];
        auto current_token_count = expert_offset[i + 1] - current_token_offset;
        auto workspace_byte = reinterpret_cast<char*>(workspace);
        auto current_idx = i % mMaxConcurrency;
        auto next_idx = (i + 1) % mMaxConcurrency;
        auto current_stream = mStreams[current_idx];
        auto next_stream = mStreams[next_idx];
        auto current_workspace = workspace_byte + mSublayerWorkspacecSize * current_idx;
        auto next_workspace = workspace_byte + mSublayerWorkspacecSize * next_idx;
        CUDA_SAFE_CALL(cudaStreamSynchronize(next_stream));
        // start copying weights to next expert
        if (i != mExpertCount - 1) {
            mSublayer->copyWeights(next_workspace, i + 1, next_stream);
        }
        // run expert on corresponding input / output buffer
        CUBLAS_SAFE_CALL(cublasSetStream_v2(mCublasHandle, current_stream));
        assert(mSublayer->run(current_token_count, current_workspace, d_routed_features + current_token_offset,
                              d_post_expert_features + current_token_offset,
                              current_workspace + mSublayer->weightSize(), current_stream));
    }

    // 5. free CPU buffer & synchronize all streams
    delete[] expert_offset;
    for (int i = 0; i < mMaxConcurrency; ++i) {
        CUDA_SAFE_CALL(cudaStreamSynchronize(mStreams[i]));
    }
    CUBLAS_SAFE_CALL(cublasSetStream_v2(mCublasHandle, stream));

    // TODO: support dynamic switching method
    // 6. mix features before & after expert
    // 7. unshuffle results
    moe_expert_base_layer_fused_mix_and_gather(token_num, token_len, d_token_pos, d_routed_features,
                                               d_post_expert_features, d_routed_mix_coeff, d_layer_output, stream);

    // 7. unshuffle results
    // moe_expert_gather(token_num, token_len, d_routed_features, d_token_pos, d_layer_output, stream);

    return 0;
}

size_t MoELayerPlugin::getSerializationSize() const noexcept {
    return METADATA_LENGTH + mExpertCentroidsCPU.count * sizeof(float) + strlen(mExpertWeightFile) + strlen(mSublayerType);
}

void MoELayerPlugin::serialize(void* buffer) const noexcept {
    // 4 int32_t
    auto int_buffer = reinterpret_cast<int*>(buffer);
    *int_buffer++ = mExpertCount;
    *int_buffer++ = mHiddenSize;
    *int_buffer++ = mMaxConcurrency;
    auto expert_weight_file_len = strlen(mExpertWeightFile);
    auto sublayer_type_len = strlen(mSublayerType);
    *int_buffer++ = expert_weight_file_len;
    *int_buffer++ = sublayer_type_len;
    // 2 strings
    auto char_buffer = reinterpret_cast<char*>(int_buffer);
    strcpy(char_buffer, mExpertWeightFile);
    char_buffer += expert_weight_file_len + 1;
    strcpy(char_buffer, mSublayerType);
    char_buffer += sublayer_type_len + 1;
    auto string_size = expert_weight_file_len + sublayer_type_len + 2;
    // align to 8 byte
    if (string_size % 8 != 0) {
        char_buffer += 8 - (string_size % 8);
    }
    // int64_t
    auto int64_buffer = reinterpret_cast<int64_t*>(char_buffer);
    *int64_buffer++ = mExpertCentroidsCPU.count;
    memcpy(int64_buffer, mExpertCentroidsCPU.values, mExpertCentroidsCPU.count * sizeof(float));
}

void MoELayerPlugin::destroy() noexcept {
    dbg(this, "call destory");
    // delete this;
}

IPluginV2* MoELayerPlugin::clone() const noexcept {
    dbg("call clone");
    return new MoELayerPlugin(*this);
}

void MoELayerPlugin::setPluginNamespace(const char* pluginNamespace) noexcept { mPluginNamespace = pluginNamespace; }

const char* MoELayerPlugin::getPluginNamespace() const noexcept { return mPluginNamespace; }
