#include <sys/stat.h>
#include <unistd.h>

#include <cassert>
#include <cstdio>
#include <cstring>

#include "MoELayerPlugin.h"
#include "thirdparty/dbg.h"

REGISTER_TENSORRT_PLUGIN(MoELayerPluginCreator);

// parameter fields
namespace field_name {
const char *EXPERT_COUNT{"expert_count"};
const char *HIDDEN_SIZE{"hidden_size"};
const char *MAX_CONCURRENCY{"max_concurrency"};
const char *EXPERT_CENTROIDS{"expert_centroids"};
const char *EXPERT_WEIGHT_FILE{"expert_weight_file"};
const char *EXPERT_SUBLAYER_TYPE{"expert_sublayer_type"};
}  // namespace

// static class member
const std::array<PluginField, 6> MoELayerPluginCreator::mPluginAttributes{
    // count of experts
    PluginField{field_name::EXPERT_COUNT, nullptr, PluginFieldType::kINT32, 1},
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
};

const PluginFieldCollection MoELayerPluginCreator::mFC{MoELayerPluginCreator::mPluginAttributes.size(),
                                                       MoELayerPluginCreator::mPluginAttributes.data()};

MoELayerPluginCreator::MoELayerPluginCreator() : mPluginNamespace("") {
    dbg("initialize MoELayerPluginCreator");
}

MoELayerPluginCreator::~MoELayerPluginCreator() {}

const PluginFieldCollection *MoELayerPluginCreator::getFieldNames() noexcept { return &mFC; }

IPluginV2 *MoELayerPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) noexcept {

    dbg("invoke createPlugin with name", name);

    int expert_count = -1;
    int hidden_size = -1;
    int max_concurrency = 2;
    Weights expert_centroids;
    char *weight_file = nullptr;
    char *sublayer = nullptr;

    // parse parameters from fc
    for (int i = 0; i < fc->nbFields; ++i) {
        auto &field = fc->fields[i];
        auto name = field.name;
        if (strcmp(name, field_name::EXPERT_COUNT) == 0) {
            assert(field.length == 1 && field.data != nullptr);
            expert_count = *static_cast<const int *>(field.data);
        } else if (strcmp(name, field_name::HIDDEN_SIZE) == 0) {
            assert(field.length == 1 && field.data != nullptr);
            hidden_size = *static_cast<const int *>(field.data);
        } else if (strcmp(name, field_name::MAX_CONCURRENCY) == 0) {
            assert(field.length == 1 && field.data != nullptr);
            max_concurrency = *static_cast<const int *>(field.data);
        } else if (strcmp(name, field_name::EXPERT_CENTROIDS) == 0) {
            assert(field.length > 0 && field.data != nullptr);
            auto centroids = new float[field.length];
            memcpy(centroids, field.data, field.length * sizeof(float));
            expert_centroids.type = DataType::kFLOAT;
            expert_centroids.count = field.length;
            expert_centroids.values = centroids;
        } else if (strcmp(name, field_name::EXPERT_WEIGHT_FILE) == 0) {
            assert(field.length > 0 && field.data != nullptr);
            weight_file = strdup(static_cast<const char *>(field.data));
        } else if (strcmp(name, field_name::EXPERT_SUBLAYER_TYPE) == 0) {
            dbg(static_cast<const char*>(field.data));
            assert(field.length > 0 && field.data != nullptr);
            sublayer = strdup(static_cast<const char *>(field.data));
        } else {
            fprintf(stderr, "unknown field name in PluginFieldCollection: %s\n", name);
            assert(false);
        }
    }

    // check parameters
    assert(expert_count > 0);
    assert(hidden_size > 0);
    assert(max_concurrency > 0);
    assert(expert_centroids.values != nullptr);
    assert(sublayer != nullptr);

    dbg(expert_count, hidden_size, max_concurrency, expert_centroids.count, sublayer, weight_file);

    struct stat64 weight_stat {};
    if (stat64(weight_file, &weight_stat) != 0) {
        perror("Cannot stat() weight file");
        assert(false);
    }
    if (!S_ISREG(weight_stat.st_mode)) {
        fprintf(stderr, "weight file must be a file\n");
        assert(false);
    }

    auto plugin = new MoELayerPlugin(name, expert_count, hidden_size, max_concurrency, expert_centroids, weight_file, sublayer);
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
