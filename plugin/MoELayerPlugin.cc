#include "MoELayerPlugin.h"

#include <cublas_v2.h>
#include <stdio.h>

#include "cuda/moe.h"
#include "cuda/ops.h"
#include "sublayers/IdentityLayer.hh"
#include "sublayers/T5FFLayer.h"
#include "thirdparty/dbg.h"
#include "utility.h"


void MoELayerPlugin::ensureGPUWeights() {
    if (mCentroidsGpu != nullptr) return;
    dbg("first time copy weights to GPU");
    auto size = centroidsSize() * sizeof(float);
    CUDA_SAFE_CALL(cudaMalloc(&mCentroidsGpu, size));
    CUDA_SAFE_CALL(cudaMemcpy(mCentroidsGpu, mCentroidsCpu, size, cudaMemcpyHostToDevice));
    if (mLayernormCpu != nullptr) {
        dbg("copy layer norm weights additionally");
        auto size = mEmbeddingSize * sizeof(float);
        CUDA_SAFE_CALL(cudaMalloc(&mLayernormGpu, size));
        CUDA_SAFE_CALL(cudaMemcpy(mLayernormGpu, mLayernormCpu, size, cudaMemcpyHostToDevice));
    }
}

void MoELayerPlugin::createSublayer() {
    assert(mSublayerType != nullptr);
    assert(mSublayer.get() == nullptr);
    // initialize sublayer according to parameter
    if (strcmp(mSublayerType, sublayer_type::T5FF) == 0) {
        mSublayer =
            std::make_shared<T5FFLayer>(mExpertCount, mEmbeddingSize, mHiddenSize, mExpertWeightFile, mMaxConcurrency);
    } else if (strcmp(mSublayerType, sublayer_type::Identity) == 0) {
        mSublayer = std::make_shared<IdentityLayer>();
    } else {
        fprintf(stderr, "ERROR: unsupported sublayer type: %s\n", mSublayerType);
        assert(false);
    }
}

// static function
MoEFlags MoELayerPlugin::parseFlags(const char* moeVariant) {
    MoEFlags flags;
    assert(moeVariant != nullptr);
    if (strcmp(moeVariant, moe_variant::BASE_LAYER) == 0) {
        flags.baseLayerOutputMix = true;
    } else if (strcmp(moeVariant, moe_variant::CPM_2) == 0) {
        flags.layernormOnInputBeforeScore = true;
    } else if (strcmp(moeVariant, moe_variant::DEFAULT) == 0) {
        // do nothing
    } else {
        fprintf(stderr, "ERROR: unsupported moe variant: %s\n", moeVariant);
        assert(false);
    }
    return flags;
}

MoELayerPlugin::MoELayerPlugin(const char* layerName, int expertCount, int embeddingSize, int hiddenSize,
                               int maxConcurrency, float* centroidsCpu, float* layernormCpu,
                               const char* expertWeightFile, const char* sublayerType, const MoEFlags flags)
    : mLayerName(strdup(layerName)),
      mExpertCount(expertCount),
      mEmbeddingSize(embeddingSize),
      mHiddenSize(hiddenSize),
      mMaxConcurrency(maxConcurrency),
      mCentroidsCpu(centroidsCpu),
      mLayernormCpu(layernormCpu),
      mExpertWeightFile(expertWeightFile),
      mSublayerType(strdup(sublayerType)),
      mFlags(flags) {
    dbg(this, "MoELayerPlugin main constructor");
    // check parameters
    assert(mCentroidsCpu != nullptr);
    if (mFlags.layernormOnInputBeforeScore && layernormCpu == nullptr) {
        fprintf(stderr, "ERROR: might provide layer norm weight if layernormOnInputBeforeScore is set\n");
        assert(false);
    }
    createSublayer();
}

MoELayerPlugin::MoELayerPlugin(const MoELayerPlugin& src)
    : MoELayerPlugin(strdup(src.mLayerName), src.mExpertCount, src.mEmbeddingSize, src.mHiddenSize, src.mMaxConcurrency,
                     src.mCentroidsCpu, src.mLayernormCpu, strdup(src.mExpertWeightFile), strdup(src.mSublayerType),
                     src.mFlags) {
    dbg(this, "MoELayerPlugin copy constructor");
    dbg(centroidsSize(), src.mExpertCount, src.mEmbeddingSize);
    // WORKAROUND
    mSublayer = nullptr;
    // copy centroids
    size_t size = centroidsSize();
    mCentroidsCpu = new float[size];
    memcpy(mCentroidsCpu, src.mCentroidsCpu, size * sizeof(float));
    // copy layer norm
    if (src.mLayernormCpu != nullptr) {
        mLayernormCpu = new float[mEmbeddingSize];
        memcpy(mLayernormCpu, src.mLayernormCpu, mEmbeddingSize * sizeof(float));
    }
    this->mSublayer = src.mSublayer;
    // createSublayer();
}

MoELayerPlugin::MoELayerPlugin(const char* layerName, const void* serialData, size_t serialLength)
    : mLayerName(strdup(layerName)) {
    dbg(this, "construct MoELayerPlugin from serialized data");
    assert(serialLength >= METADATA_LENGTH);
    // 6 int32_t
    auto int_buffer = reinterpret_cast<const int*>(serialData);
    mExpertCount = *int_buffer++;
    mEmbeddingSize = *int_buffer++;
    mHiddenSize = *int_buffer++;
    mMaxConcurrency = *int_buffer++;
    auto expert_weight_file_len = *int_buffer++;
    auto sublayer_type_len = *int_buffer++;
    // flag
    auto flag_buffer = reinterpret_cast<const MoEFlags*>(int_buffer);
    mFlags = *flag_buffer++;
    // 2 strings
    auto char_buffer = reinterpret_cast<const char*>(flag_buffer);
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
    auto float_buffer = reinterpret_cast<const int64_t*>(char_buffer);
    auto size = centroidsSize();
    // assert(size * sizeof(float) == serialLength - METADATA_LENGTH);
    mCentroidsCpu = new float[size];
    memcpy(mCentroidsCpu, float_buffer, size * sizeof(float));
    float_buffer += size;
    // initialize layer norm
    if (mFlags.layernormOnInputBeforeScore) {
        mLayernormCpu = new float[mEmbeddingSize];
        memcpy(mLayernormCpu, float_buffer, size * sizeof(float));
    }
    createSublayer();
}

MoELayerPlugin::~MoELayerPlugin() {
    dbg(this, "destructing MoELayerPlugin");
    terminate();
}

DimsExprs MoELayerPlugin::getOutputDimensions(int32_t outputIndex, const DimsExprs* inputs, int32_t nbInputs,
                                              IExprBuilder& exprBuilder) noexcept {
    dbg(outputIndex, nbInputs);
    assert(outputIndex == 0);
    return mSublayer->getOutputDimensions(inputs, exprBuilder);
}

bool MoELayerPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs,
                                               int32_t nbOutputs) noexcept {
    assert(nbInputs == 1 && nbOutputs == 1);
    auto& desc = inOut[pos];
    return desc.format == TensorFormat::kLINEAR && desc.type == DataType::kFLOAT;
}

nvinfer1::DataType MoELayerPlugin::getOutputDataType([[maybe_unused]] int32_t index,
                                                     nvinfer1::DataType const* inputTypes,
                                                     [[maybe_unused]] int32_t nbInputs) const noexcept {
    return inputTypes[0];
}

void MoELayerPlugin::configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInputs,
                                     const DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept {
    assert(nbInputs == 1 && nbOutputs == 1);
    dbg(in[0].desc.dims.d);
    assert(mSublayer->configureWithFormat(&in[0].desc.dims, nbInputs, &out[0].desc.dims, nbOutputs));
    auto& dim = in[0].desc.dims;
    assert(dim.nbDims == 3);
    assert(mEmbeddingSize == dim.d[2]);
    mSequenceLength = dim.d[1];
    dbg(mEmbeddingSize, mSequenceLength);
}

void MoELayerPlugin::ensureCUDAContext() {
    if (mStreams == nullptr) {
        dbg("first time create CUDA streams");
        mStreams = new cudaStream_t[mMaxConcurrency];
        for (int i = 0; i < mMaxConcurrency; ++i) {
            CUDA_SAFE_CALL(cudaStreamCreate(&mStreams[i]));
        }
    }
    if (mCublasHandle == nullptr) {
        CUBLAS_SAFE_CALL(cublasCreate_v2(&mCublasHandle));
        assert(mCublasHandle != nullptr);
        mSublayer->setCuBlasHandle(mCublasHandle);
    }
}

int32_t MoELayerPlugin::initialize() noexcept {
    dbg(this, "call initialize");
    mSublayer->initialize();
    return 0;
}

void MoELayerPlugin::terminate() noexcept {
    dbg(this, "call terminate");
    // free centroids on CPU and GPU
    if (mCentroidsCpu != nullptr) {
        delete[] mCentroidsCpu;
        mCentroidsCpu = nullptr;
    }
    if (mCentroidsGpu != nullptr) {
        CUDA_SAFE_CALL(cudaFree(mCentroidsGpu));
        mCentroidsGpu = nullptr;
    }
    // free layer norm
    if (mLayernormCpu != nullptr) {
        delete[] mLayernormCpu;
        mLayernormCpu = nullptr;
    }
    if (mLayernormGpu != nullptr) {
        CUDA_SAFE_CALL(cudaFree(mLayernormGpu));
        mLayernormGpu = nullptr;
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
    // decrement sublayer ref counter
    mSublayer.reset();
}

void MoELayerPlugin::ensureSublayerWorkspaceSize(size_t tokenCount) const {
    mSublayerWorkspacecSize = mSublayer->weightSize() + mSublayer->workspaceSize(tokenCount);
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
size_t MoELayerPlugin::getWorkspaceSize(const PluginTensorDesc* inputs, int32_t nbInputs,
                                        [[maybe_unused]] const PluginTensorDesc* outputs,
                                        int32_t nbOutputs) const noexcept {
    // the maximum tokens that might go to one single expert
    // FIXME: currently set to full size
    assert(nbInputs == 1 && nbOutputs == 1 && inputs[0].dims.nbDims == 3);
    auto& input_dim = inputs[0].dims;
    dbg(input_dim.d);
    size_t batch_size = input_dim.d[0];
    auto max_single_expert_token_count = static_cast<int32_t>(batch_size * mSequenceLength);
    ensureSublayerWorkspaceSize(max_single_expert_token_count);
    auto sublayer_size = mSublayerWorkspacecSize * mMaxConcurrency;
    // maximum tokens that might be processed by this layer
    auto max_token_count = batch_size * mSequenceLength;
    auto plugin_size =
        (max_token_count * mExpertCount + max_token_count * 2 + max_token_count * mEmbeddingSize * 2) * sizeof(float) +
        max_token_count * 2 * sizeof(int);
    auto final_size = plugin_size + sublayer_size;
    dbg(final_size);
    return final_size;
}

namespace {
static cudaDeviceProp DEVICE_PROP = cudaDevicePropDontCare;
}

int32_t MoELayerPlugin::enqueue(const PluginTensorDesc* inputDesc, [[maybe_unused]] const PluginTensorDesc* outputDesc,
                                const void* const* inputs, void* const* outputs, void* workspace,
                                cudaStream_t stream) noexcept {
    // dbg(batchSize);
    // run the actual MoE calculation
    // 0. obtain all buffers
    dbg(this);
    ensureCUDAContext();
    ensureGPUWeights();
    auto batch_size = inputDesc[0].dims.d[0];
    auto token_num = batch_size * mSequenceLength;
    auto token_len = mEmbeddingSize;
    ensureSublayerWorkspaceSize(token_num);
    // dbg(token_num, token_len);
    auto d_layer_input = static_cast<const float*>(inputs[0]);
    auto d_expert_centroids = static_cast<const float*>(mCentroidsGpu);
    auto d_layer_norm_weights = static_cast<const float*>(mLayernormGpu);
    auto moe_buffer =
        reinterpret_cast<float*>(static_cast<char*>(workspace) + mSublayerWorkspacecSize * mMaxConcurrency);
    auto d_token_expert_aff = moe_buffer;
    auto d_gate_selection = reinterpret_cast<int*>(moe_buffer + token_num * mExpertCount);
    auto d_token_pos = d_gate_selection + token_num;
    auto d_routed_features = reinterpret_cast<float*>(d_token_pos + token_num);
    auto d_post_expert_features = d_routed_features + token_num * token_len;
    auto d_mix_coeff = d_post_expert_features + token_num * token_len;
    auto d_routed_mix_coeff = d_mix_coeff + token_num;
    auto d_layer_output = static_cast<float*>(outputs[0]);

    CHECK_CUDA_POINTER(d_layer_output);
    CHECK_CUDA_POINTER(d_mix_coeff);
    CHECK_CUDA_POINTER(d_post_expert_features);
    CHECK_CUDA_POINTER(d_routed_features);

    const float* d_affiliation_input = d_layer_input;

    // 0. pre-process input if needed
    if (mFlags.layernormOnInputBeforeScore) {
        dbg("run layernorm on input");
        CHECK_CUDA_POINTER(d_layer_norm_weights);
        if (DEVICE_PROP.major == -1) {
            CUDA_SAFE_CALL(cudaGetDeviceProperties(&DEVICE_PROP, 0));
            assert(DEVICE_PROP.major >= 6);  // we don't want too old devices
        }
        dbg(DEVICE_PROP.maxGridSize);
        // temporarily use d_routed_features to store input after layernorm
        layernorm_gpu<float, float>(d_routed_features, d_layer_input, token_num, mEmbeddingSize, (double)1e-6,
                                    d_layer_norm_weights, nullptr, DEVICE_PROP.maxGridSize[1], stream);
        d_affiliation_input = d_routed_features;
    }

    // 1. calculate token-expert affiliation
    // (token_num, token_len) @ (token_len, expert_count)
    float alpha = 1.0, beta = 0.0;
    CUBLAS_SAFE_CALL(cublasSetStream_v2(mCublasHandle, stream));
    CUBLAS_SAFE_CALL(cublasSgemm_v2(mCublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, mExpertCount, token_num, token_len, &alpha,
                                    d_expert_centroids, token_len, d_affiliation_input, token_len, &beta,
                                    d_token_expert_aff, mExpertCount));
    CUDA_SAFE_CALL(cudaStreamSynchronize(stream));

    // dbg("after affiliation");
    // showCudaArray(d_layer_input, token_num, token_len);
    // showArray(static_cast<const float*>(mExpertCentroidsCPU.values), mExpertCount, token_len);
    // showCudaArray(d_expert_centroids, mExpertCount, token_len);
    // showCudaArray(d_token_expert_aff, token_num, mExpertCount);

    // 2. get expert assignments (TODO: support multiple experts for each token)
    moe_expert_select(token_num, mExpertCount, d_token_expert_aff, d_gate_selection, d_mix_coeff, stream);
    // dbg("after select");
    // showCudaArray(d_mix_coeff, 1, token_num);

    // 3. count & sort & gather (a.k.a. shuffle) tokens for each expert
    auto expert_offset = new int[mExpertCount + 1](), expert_count = new int[mExpertCount]();
    expert_offset[mExpertCount] = token_num;
    moe_expert_count(token_num, mExpertCount, d_gate_selection, d_token_pos, expert_count, expert_offset, stream);
    // dbg("after count");
    moe_expert_scatter(token_num, token_len, d_layer_input, d_mix_coeff, d_token_pos, d_routed_features,
                       d_routed_mix_coeff, stream);
    // dbg("after scatter");
    // showCudaArray(d_routed_features, token_num, token_len);
    // showCudaArray(d_routed_mix_coeff, 1, token_num);

    // 4. run each expert: state = sublayer.run(state) (skip expert with empty data)
    int next_expert = 0;  // skip all expert with no work to do
    while (expert_count[next_expert] == 0 && next_expert < mExpertCount) next_expert++;
    assert(next_expert < mExpertCount);

    mSublayer->copyWeights(workspace, next_expert, mStreams[0]);
    // dbg("after first copy");

    CUDA_SAFE_CALL(cudaStreamSynchronize(stream));

    // i: expert index, j: expert (with non-empty features) index
    for (int i = next_expert, j = 0; i < mExpertCount; i = next_expert, j++) {
        auto current_token_offset = expert_offset[i];
        auto workspace_byte = reinterpret_cast<char*>(workspace);
        auto current_idx = j % mMaxConcurrency;
        auto next_idx = (j + 1) % mMaxConcurrency;
        auto current_stream = mStreams[current_idx];
        auto next_stream = mStreams[next_idx];
        auto current_workspace = workspace_byte + mSublayerWorkspacecSize * current_idx;
        auto next_workspace = workspace_byte + mSublayerWorkspacecSize * next_idx;
        CUDA_SAFE_CALL(cudaStreamSynchronize(next_stream));
        // start copying weights to next expert
        next_expert = i + 1;
        while (expert_count[next_expert] == 0 && next_expert < mExpertCount) next_expert++;
        if (next_expert < mExpertCount) {
            mSublayer->copyWeights(next_workspace, next_expert, next_stream);
        }
        // run expert on corresponding input / output buffer
        CUBLAS_SAFE_CALL(cublasSetStream_v2(mCublasHandle, current_stream));
        dbg(i);
        mSublayer->run(expert_count[i], current_workspace, d_routed_features + current_token_offset,
                       d_post_expert_features + current_token_offset, current_workspace + mSublayer->weightSize(),
                       current_stream);
    }

    // 5. free CPU buffer & synchronize all streams
    delete[] expert_offset;
    delete[] expert_count;
    for (int i = 0; i < mMaxConcurrency; ++i) {
        CUDA_SAFE_CALL(cudaStreamSynchronize(mStreams[i]));
    }
    CUBLAS_SAFE_CALL(cublasSetStream_v2(mCublasHandle, stream));

    // 6. (optional) mix features before & after expert
    // 7. unshuffle results
    // dbg("before gather");
    if (mFlags.baseLayerOutputMix) {
        moe_expert_base_layer_fused_mix_and_gather(token_num, token_len, d_token_pos, d_routed_features,
                                                   d_post_expert_features, d_routed_mix_coeff, d_layer_output, stream);
    } else {
        moe_expert_gather(token_num, token_len, d_routed_features, d_token_pos, d_layer_output, stream);
    }
    // showCudaArray(d_layer_output, token_num, token_len);

    return 0;
}

size_t MoELayerPlugin::getSerializationSize() const noexcept {
    auto total_size =
        METADATA_LENGTH + strlen(mExpertWeightFile) + strlen(mSublayerType) + centroidsSize() * sizeof(float);
    if (mLayernormCpu != nullptr) total_size += sizeof(float) * mEmbeddingSize;
    return total_size;
}

void MoELayerPlugin::serialize(void* buffer) const noexcept {
    // 6 int32_t
    auto int_buffer = reinterpret_cast<int*>(buffer);
    *int_buffer++ = mExpertCount;
    *int_buffer++ = mEmbeddingSize;
    *int_buffer++ = mHiddenSize;
    *int_buffer++ = mMaxConcurrency;
    auto expert_weight_file_len = strlen(mExpertWeightFile);
    auto sublayer_type_len = strlen(mSublayerType);
    *int_buffer++ = expert_weight_file_len;
    *int_buffer++ = sublayer_type_len;
    // flag
    auto flag_buffer = reinterpret_cast<MoEFlags*>(int_buffer);
    *flag_buffer++ = mFlags;
    // 2 strings
    auto char_buffer = reinterpret_cast<char*>(flag_buffer);
    strcpy(char_buffer, mExpertWeightFile);
    char_buffer += expert_weight_file_len + 1;
    strcpy(char_buffer, mSublayerType);
    char_buffer += sublayer_type_len + 1;
    auto string_size = expert_weight_file_len + sublayer_type_len + 2;
    // align to 8 byte
    if (string_size % 8 != 0) {
        char_buffer += 8 - (string_size % 8);
    }
    // centroids
    auto float_buffer = reinterpret_cast<int64_t*>(char_buffer);
    memcpy(float_buffer, mCentroidsCpu, centroidsSize() * sizeof(float));
    float_buffer += centroidsSize();
    // layer norm
    if (mLayernormCpu != nullptr) {
        memcpy(float_buffer, mLayernormCpu, mEmbeddingSize * sizeof(float));
    }
}

void MoELayerPlugin::destroy() noexcept {
    dbg(this, "call destory");
    // delete this;
}

IPluginV2DynamicExt* MoELayerPlugin::clone() const noexcept {
    dbg("call clone");
    return new MoELayerPlugin(*this);
}

void MoELayerPlugin::setPluginNamespace(const char* pluginNamespace) noexcept { mPluginNamespace = pluginNamespace; }

const char* MoELayerPlugin::getPluginNamespace() const noexcept { return mPluginNamespace; }
