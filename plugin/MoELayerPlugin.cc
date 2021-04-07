#include "MoELayerPlugin.h"

#include <cublas.h>
#include <stdio.h>

REGISTER_TENSORRT_PLUGIN(MoELayerPluginCreator);

MoELayerPlugin::MoELayerPlugin() { printf("Hello World\n"); }
