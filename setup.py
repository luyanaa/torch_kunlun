import os
from pathlib import Path

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

ROOT = Path(__file__).resolve().parent.parent

XRE = Path(os.environ.get("KUNLUN_XRE", ROOT / "xre-ubuntu_x86_64"))
XDNN = Path(os.environ.get("KUNLUN_XDNN", ROOT / "xdnn-ubuntu_x86_64"))
XCCL = Path(os.environ.get("KUNLUN_XCCL", ROOT / "xccl-ubuntu_x86_64"))

include_dirs = [
    str(XRE / "include"),
    str(XDNN / "include"),
    str(XCCL / "include"),
]
library_dirs = [
    str(XRE / "so"),
    str(XDNN / "so"),
    str(XCCL / "so"),
]

extra_compile_args = [
    "-std=c++17",
    "-O2",
    "-fPIC",
    f"-D_GLIBCXX_USE_CXX11_ABI={int(getattr(torch._C, '_GLIBCXX_USE_CXX11_ABI', 1))}",
]

extra_link_args = [
    f"-Wl,-rpath,{XRE / 'so'}",
    f"-Wl,-rpath,{XDNN / 'so'}",
    f"-Wl,-rpath,{XCCL / 'so'}",
]

setup(
    name="torch_kunlun",
    version="0.1.0",
    description="PyTorch PrivateUse1 runtime for Kunlun K200 (XPU1) via XDNN",
    packages=find_packages(),
    ext_modules=[
        CppExtension(
            name="torch_kunlun._C",
            sources=[
                "csrc/kunlun.cpp",
                "csrc/allocator.cpp",
                "csrc/ops.cpp",
                "csrc/ops_nn.cpp",
                "csrc/ops_nn_extra.cpp",
                "csrc/ops_extra.cpp",
                "csrc/autograd.cpp",
                "csrc/autograd_extra.cpp",
                "csrc/autograd_nn_extra.cpp",
                "csrc/linalg_extra.cpp",
                "csrc/einsum_extra.cpp",
                "csrc/optim_extra.cpp",
                "csrc/loss_extra.cpp",
                "csrc/xccl.cpp",
            ],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=["xpuapi", "xpurt", "bkcl"],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch>=2.6"],
    python_requires=">=3.9",
    zip_safe=False,
)
