import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="copy_and_cat_engine",
    version="0.2",
    packages=["copy_and_cat_engine"],
    ext_modules=[
        CUDAExtension(
            name="copy_and_cat_engine",
            sources=[
                "copy_and_cat_engine/copy_and_cat_engine.cu",
                "copy_and_cat_engine/python_api.cpp",
            ],
            # extra_compile_args={"nvcc": ["--use_fast_math"]},
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }, 
)