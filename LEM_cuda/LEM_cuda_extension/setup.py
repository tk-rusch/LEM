from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='lem_cuda',
    ext_modules=[
        CUDAExtension('lem_cuda', [
            'lem_cuda.cpp',
            'lem_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
