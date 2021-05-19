#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cassert>
#include <cstring>

#include "MoELayerPlugin.h"


REGISTER_TENSORRT_PLUGIN(MoELayerPluginCreator);

// parameter fields
namespace {
const char *FIELD_EXPERT_COUNT{"expert_count"};
const char *FIELD_HIDDEN_SIZE{"hidden_size"};
const char *FIELD_EXPERT_CENTROIDS{"expert_centroids"};
const char *FIELD_EXPERT_WEIGHT_FILE{"expert_weight_file"};
}  // namespace

// static class member
const std::array<PluginField, 4> MoELayerPluginCreator::mPluginAttributes{
    // count of experts
    PluginField{FIELD_EXPERT_COUNT, nullptr, PluginFieldType::kINT32, 1},
    // DIM -> hidden_size -> DIM
    PluginField{FIELD_HIDDEN_SIZE, nullptr, PluginFieldType::kINT32, 1},
    // mapping of token to expert
    PluginField{FIELD_EXPERT_CENTROIDS, nullptr, PluginFieldType::kFLOAT32, 1},
    // weight of experts, read from separate files
    PluginField{FIELD_EXPERT_WEIGHT_FILE, nullptr, PluginFieldType::kUNKNOWN, 1},
};

const PluginFieldCollection MoELayerPluginCreator::mFC{MoELayerPluginCreator::mPluginAttributes.size(),
                                                       MoELayerPluginCreator::mPluginAttributes.data()};

MoELayerPluginCreator::MoELayerPluginCreator() : mPluginNamespace("UNKNOWN") {}

MoELayerPluginCreator::~MoELayerPluginCreator() {}

const PluginFieldCollection *MoELayerPluginCreator::getFieldNames() noexcept { return &mFC; }

IPluginV2 *MoELayerPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) noexcept {

    int expert_count = -1;
    int hidden_size = -1;
    Weights expert_centroids;
    char *weight_file = nullptr;

    // parse parameters from fc
    for (int i = 0; i < fc->nbFields; ++i) {
        auto &field = fc->fields[i];
        auto name = field.name;
        if (strcmp(name, FIELD_EXPERT_COUNT) == 0) {
            assert(field.length == 1 && field.data != nullptr);
            expert_count = *static_cast<const int *>(field.data);
        } else if (strcmp(name, FIELD_HIDDEN_SIZE) == 0) {
            assert(field.length == 1 && field.data != nullptr);
            hidden_size = *static_cast<const int *>(field.data);
        } else if (strcmp(name, FIELD_EXPERT_CENTROIDS) == 0) {
            assert(field.length > 0 && field.data != nullptr);
            auto centroids = static_cast<float *>(malloc(sizeof(float) * field.length));
            memcpy(centroids, field.data, field.length * sizeof(float));
            expert_centroids.type = DataType::kFLOAT;
            expert_centroids.count = field.length;
            expert_centroids.values = centroids;
        } else if (strcmp(name, FIELD_EXPERT_WEIGHT_FILE) == 0) {
            assert(field.length == 1 && field.data != nullptr);
            weight_file = strdup(static_cast<const char *>(field.data));
        } else {
            fprintf(stderr, "unknown field name in PluginFieldCollection: %s\n", name);
            assert(false);
        }
    }

    // check parameters
    assert(expert_count > 0);
    assert(hidden_size > 0);
    assert(expert_centroids.values != nullptr);

    struct stat64 weight_stat {};
    if (stat64(weight_file, &weight_stat) != 0) {
        perror("Cannot stat() weight file: ");
        exit(1);
    }
    if (!S_ISREG(weight_stat.st_mode)) {
        fprintf(stderr, "weight file must be a file\n");
        exit(1);
    }

    auto plugin = new MoELayerPlugin(name, expert_count, hidden_size, expert_centroids, weight_file);
    plugin->setPluginNamespace(mPluginNamespace);

    return plugin;
}

IPluginV2 *MoELayerPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept {
    auto plugin = new MoELayerPlugin(name, serialData, serialLength);
    plugin->setPluginNamespace(mPluginNamespace);
    return plugin;
}

void MoELayerPluginCreator::setPluginNamespace(const char *pluginNamespace) noexcept { mPluginNamespace = pluginNamespace; }

const char *MoELayerPluginCreator::getPluginNamespace() const noexcept { return mPluginNamespace; }
