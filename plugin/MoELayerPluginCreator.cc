#include <sys/stat.h>
#include <unistd.h>

#include <cassert>
#include <cstdio>
#include <cstring>

#include "MoELayerPlugin.h"
#include "thirdparty/dbg.h"
#include "utility.h"

REGISTER_TENSORRT_PLUGIN(MoELayerPluginCreator);

// parameter fields
namespace field_name {
const char *EXPERT_COUNT{"expert_count"};
const char *EMBEDDING_SIZE{"embedding_size"};
const char *HIDDEN_SIZE{"hidden_size"};
const char *MAX_CONCURRENCY{"max_concurrency"};
const char *EXPERT_CENTROIDS{"expert_centroids"};
const char *EXPERT_WEIGHT_FILE{"expert_weight_file"};
const char *EXPERT_SUBLAYER_TYPE{"expert_sublayer_type"};
const char *MOE_VARIANT{"moe_variant"};
const char *LAYERNORM_WEIGHT{"layernorm_weight"};
}  // namespace field_name

// static class member
const std::array<PluginField, 9> MoELayerPluginCreator::mPluginAttributes{
    // count of experts
    PluginField{field_name::EXPERT_COUNT, nullptr, PluginFieldType::kINT32, 1},
    // embedding size
    PluginField{field_name::EMBEDDING_SIZE, nullptr, PluginFieldType::kINT32, 1},
    // DIM -> hidden_size -> DIM
    PluginField{field_name::HIDDEN_SIZE, nullptr, PluginFieldType::kINT32, 1},
    // max concurrent experts in GPU memory
    PluginField{field_name::MAX_CONCURRENCY, nullptr, PluginFieldType::kINT32, 1},
    // mapping of token to expert
    PluginField{field_name::EXPERT_CENTROIDS, nullptr, PluginFieldType::kFLOAT32, 1},
    // weight of experts, read from separate files
    PluginField{field_name::EXPERT_WEIGHT_FILE, nullptr, PluginFieldType::kUNKNOWN, 1},
    // type of expert sub-layer
    PluginField{field_name::EXPERT_SUBLAYER_TYPE, sublayer_type::T5FF, PluginFieldType::kUNKNOWN, 1},
    // type of MoE variant
    PluginField{field_name::MOE_VARIANT, moe_variant::CPM_2, PluginFieldType::kUNKNOWN, 1},
    // type of MoE variant
    PluginField{field_name::LAYERNORM_WEIGHT, nullptr, PluginFieldType::kFLOAT32, 1},
};

const PluginFieldCollection MoELayerPluginCreator::mFC{MoELayerPluginCreator::mPluginAttributes.size(),
                                                       MoELayerPluginCreator::mPluginAttributes.data()};

MoELayerPluginCreator::MoELayerPluginCreator() : mPluginNamespace("") { dbg("initialize MoELayerPluginCreator"); }

MoELayerPluginCreator::~MoELayerPluginCreator() {}

const PluginFieldCollection *MoELayerPluginCreator::getFieldNames() noexcept { return &mFC; }

IPluginV2 *MoELayerPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) noexcept {

    dbg("invoke createPlugin with name", name);

    int expert_count = -1;
    int embedding_size = -1;
    int hidden_size = -1;
    int max_concurrency = 2;
    float *expert_centroids = nullptr;
    float *layernorm_weight = nullptr;
    char *weight_file = nullptr;
    char *sublayer = nullptr;
    char *variant = nullptr;
    int centroid_length;
    int layernorm_length;

    // parse parameters from fc
    for (int i = 0; i < fc->nbFields; ++i) {
        auto &field = fc->fields[i];
        auto name = field.name;
        if (strcmp(name, field_name::EXPERT_COUNT) == 0) {
            assert(field.length == 1 && field.data != nullptr);
            expert_count = *static_cast<const int *>(field.data);
        } else if (strcmp(name, field_name::EMBEDDING_SIZE) == 0) {
            assert(field.length == 1 && field.data != nullptr);
            embedding_size = *static_cast<const int *>(field.data);
        } else if (strcmp(name, field_name::HIDDEN_SIZE) == 0) {
            assert(field.length == 1 && field.data != nullptr);
            hidden_size = *static_cast<const int *>(field.data);
        } else if (strcmp(name, field_name::MAX_CONCURRENCY) == 0) {
            assert(field.length == 1 && field.data != nullptr);
            max_concurrency = *static_cast<const int *>(field.data);
        } else if (strcmp(name, field_name::EXPERT_CENTROIDS) == 0) {
            assert(field.length > 0 && field.data != nullptr);
            centroid_length = field.length;
            expert_centroids = new float[field.length];
            memcpy(expert_centroids, field.data, field.length * sizeof(float));
        } else if (strcmp(name, field_name::EXPERT_WEIGHT_FILE) == 0) {
            assert(field.length > 0 && field.data != nullptr);
            weight_file = strdup(static_cast<const char *>(field.data));
        } else if (strcmp(name, field_name::EXPERT_SUBLAYER_TYPE) == 0) {
            dbg(static_cast<const char *>(field.data));
            assert(field.length > 0 && field.data != nullptr);
            sublayer = strdup(static_cast<const char *>(field.data));
        } else if (strcmp(name, field_name::MOE_VARIANT) == 0) {
            dbg(static_cast<const char *>(field.data));
            assert(field.length > 0 && field.data != nullptr);
            variant = strdup(static_cast<const char *>(field.data));
        } else if (strcmp(name, field_name::LAYERNORM_WEIGHT) == 0) {
            assert(field.length > 0 && field.data != nullptr);
            layernorm_length = field.length;
            layernorm_weight = new float[field.length];
            memcpy(layernorm_weight, field.data, field.length * sizeof(float));
        } else {
            fprintf(stderr, "unknown field name in PluginFieldCollection: %s\n", name);
            assert(false);
        }
    }

    // check parameters
    assert(embedding_size > 0);
    assert(expert_count > 0);
    assert(hidden_size > 0);
    assert(max_concurrency > 0);
    assert(centroid_length == embedding_size * expert_count);
    assert(expert_centroids != nullptr);
    assert(sublayer != nullptr);
    assert(variant != nullptr);
    if (layernorm_weight != nullptr) {
        assert(layernorm_length == embedding_size);
    }

    dbg(variant, expert_count, embedding_size, hidden_size, layernorm_weight, max_concurrency, sublayer, weight_file);

    struct stat64 weight_stat {};
    if (stat64(weight_file, &weight_stat) != 0) {
        perror("Cannot stat() weight file");
        assert(false);
    }
    if (!S_ISREG(weight_stat.st_mode)) {
        fprintf(stderr, "weight file must be a file\n");
        assert(false);
    }

    auto flags = MoELayerPlugin::parseFlags(variant);
    auto plugin = new MoELayerPlugin(name, expert_count, embedding_size, hidden_size, max_concurrency, expert_centroids,
                                     layernorm_weight, weight_file, sublayer, flags);
    plugin->setPluginNamespace(mPluginNamespace);

    return plugin;
}

IPluginV2 *MoELayerPluginCreator::deserializePlugin(const char *name, const void *serialData,
                                                    size_t serialLength) noexcept {
    auto plugin = new MoELayerPlugin(name, serialData, serialLength);
    plugin->setPluginNamespace(mPluginNamespace);
    return plugin;
}

void MoELayerPluginCreator::setPluginNamespace(const char *pluginNamespace) noexcept {
    mPluginNamespace = pluginNamespace;
    dbg("set plugin namespace in creator", pluginNamespace);
}

const char *MoELayerPluginCreator::getPluginNamespace() const noexcept { return mPluginNamespace; }
