project(
    'moe-infer-plugin', 'cpp', 'cuda',
    version: '0.1',
    meson_version: '>=0.50.0',
    default_options : [
      'warning_level=2',
      'cpp_std=c++17',
    ]
)

add_project_arguments('-std=c++17', language : 'cuda')
add_project_arguments('-Wno-deprecated-declarations', language : 'cpp')

cxx = meson.get_compiler('cpp')
so_ext = 'so'

# read prefix location
cudnn_prefix = get_option('WITH_CUDNN')
tensorrt_prefix = get_option('WITH_TENSORRT')

if get_option('buildtype').startswith('debug')
  add_project_arguments(['-DDEBUG', '-DDBG_MACRO_NO_WARNING'], language : ['cpp', 'cuda'])
else
  add_project_arguments(['-DNO_DEBUG', '-DDBG_MACRO_NO_WARNING', '-DDBG_MACRO_DISABLE'], language : ['cpp', 'cuda'])
endif


# find libraries
cuda_dep = dependency('cuda', version : '>=10', modules : ['cublas'])
cudnn_lib = cxx.find_library('cudnn', dirs: [cudnn_prefix / 'lib64'])
nvinfer_lib = cxx.find_library('nvinfer', dirs: [tensorrt_prefix / 'lib'])
zlib = cxx.find_library('z')
cudnn_dep = declare_dependency(dependencies: cudnn_lib)
nvinfer_dep = declare_dependency(dependencies: nvinfer_lib)
zlib_dep = declare_dependency(dependencies: zlib)

# TensorRT headers
external_inc = include_directories(
    cudnn_prefix / 'include',
    tensorrt_prefix / 'include'
)

# sources
plugin_sources = [
    'MoELayerPlugin.cc',
    'MoELayerPluginCreator.cc',
    'sublayers/T5FFLayer.cc',
    'thirdparty/cnpy/cnpy.cpp',
    'cuda/moe.cu',
    'cuda/ops/layernorm.cu',
    'cuda/ops/gelu.cu',
]

# build library
shared_library(
    'moelayer',
    plugin_sources,
    include_directories: external_inc,
    dependencies: [cuda_dep, cudnn_dep, nvinfer_dep, zlib_dep],
)
