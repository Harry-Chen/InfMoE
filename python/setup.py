import setuptools
import distutils.cmd
import os

cxx_flags = []
ext_libs = []


class build_ext(distutils.cmd.Command):

    tensorrt_prefix: str
    cudnn_prefix: str
    debug: bool

    user_options = [
        ('tensorrt-prefix=', None, 'TensorRT location'),
        ('cudnn-prefix=', None, 'cuDNN location'),
        ('debug', 'd', 'enable debug output')
    ]

    def initialize_options(self):
        self.tensorrt_prefix = '/usr'
        self.cudnn_prefix = '/usr'
        self.debug = 0

    def finalize_options(self):
        self.debug = self.debug == 1
        if not os.path.isdir(self.tensorrt_prefix):
            raise Exception(
                f'TensorRT prefix does not exist: {self.tensorrt_prefix}')
        if not os.path.isfile(os.path.join(self.tensorrt_prefix, 'include', 'NvInfer.h')):
            raise Exception(
                f'Cannot find NvInfer.h in TensorRT prefix: {self.tensorrt_prefix}')
        if not os.path.isdir(self.cudnn_prefix):
            raise Exception(
                f'cuDNN prefix does not exist: {self.cudnn_prefix}')
        if not os.path.isfile(os.path.join(self.cudnn_prefix, 'include', 'cudnn.h')):
            raise Exception(
                f'Cannot find cudnn.h in cuDNN prefix: {self.cudnn_prefix}')

    def run(self):
        import os
        import shutil
        import subprocess

        curr_dir = os.path.dirname(os.path.realpath(__file__))

        p = subprocess.Popen(['make distclean && make'], shell=True, env=dict(os.environ, **{
            'DEBUG': str(self.debug).lower(),
            'TENSORRT_PREFIX': self.tensorrt_prefix,
            'CUDNN_PREFIX': self.cudnn_prefix,
            'BUILDDIR': os.path.join(curr_dir, 'build')
        }), cwd=os.path.join(curr_dir, '..', 'plugin'))
        p.wait()
        assert p.returncode == 0, 'Build with meson failed'

        shutil.copy(
            os.path.join("build", "libtrtmoelayer.so"),
            os.path.join("trt_moe", "libtrtmoelayer.so"),
        )


if __name__ == '__main__':
    setuptools.setup(
        name='trt-moe',
        version='0.0.1',
        description='Python binding of TensorRT plugin for MoE layers',
        author='Shengqi Chen, Yanzheng Cai, Xu Han',
        author_email='shengqi.chen@tuna.tsinghua.edu.cn',
        license='Apache-2',
        url='https://github.com/Harry-Chen/trt-moe',
        packages=['trt_moe'],
        cmdclass={
            'build_ext': build_ext
        },
        package_data={
            'trt_moe': ['libtrtmoelayer.so'],
        },
    )
