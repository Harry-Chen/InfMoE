project(
    'moe-infer-plugin', 'cpp', 'cuda',
    version: '0.1',
    meson_version: '>=0.50.0',
    default_options : [
      'warning_level=3',
      'cpp_std=c++14'
    ]
)
add_global_arguments('-Wno-deprecated-declarations', language : 'cpp')

cxx = meson.get_compiler('cpp')
so_ext = 'so'

# read prefix location
cudnn_prefix = get_option('WITH_CUDNN')
tensorrt_prefix = get_option('WITH_TENSORRT')

# find libraries
cuda_dep = dependency('cuda', version : '>=10', modules : ['cublas'])
cudnn_lib = cxx.find_library('cudnn', dirs: [cudnn_prefix / 'lib64'])
nvinfer_lib = cxx.find_library('nvinfer', dirs: [tensorrt_prefix / 'lib'])
cudnn_dep = declare_dependency(dependencies: cudnn_lib)
nvinfer_dep = declare_dependency(dependencies: nvinfer_lib)

# TensorRT headers
external_inc = include_directories(
    cudnn_prefix / 'include',
    tensorrt_prefix / 'include'
)

# sources
plugin_sources = [
    'MoELayerPlugin.cc',
    'MoELayerPluginCreator.cc'
]

# build library
shared_library(
    'moelayer',
    plugin_sources,
    include_directories: external_inc,
    dependencies: [cuda_dep, cudnn_dep, nvinfer_dep]
)
