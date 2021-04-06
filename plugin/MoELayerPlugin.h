#pragma once

#ifndef MOE_LAYER_PLUGIN_H
#define MOE_LAYER_PLUGIN_H

#include <NvInferPlugin.h>

using namespace nvinfer1;

class MoELayerPlugin : public IPluginV2 {
   public:
    MoELayerPlugin();
};

class MoELayerPluginCreator : public IPluginCreator {
   public:
    MoELayerPluginCreator();
};

#endif  // MOE_LAYER_PLUGIN_H
