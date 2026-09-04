"""AmpereKV CUDA Extension 的 setuptools 构建入口。"""

from __future__ import annotations

import os
from pathlib import Path

from setuptools import setup

PROJECT_ROOT = Path(__file__).resolve().parent


def create_cuda_extension():
    """按环境变量决定是否构建 CUDA Extension。

    CPU CI 和没有 CUDA Toolkit 的本地 Windows 环境不应尝试导入
    torch.utils.cpp_extension。云端执行 make build 时会显式设置
    AMPERE_KV_BUILD_CUDA=1，从而启用下面的 CUDA 编译路径。
    """

    if os.environ.get("AMPERE_KV_BUILD_CUDA", "0") != "1":
        return [], {}

    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    except ImportError as exc:
        raise RuntimeError(
            "构建 CUDA Extension 需要先安装带 CUDA 支持的 PyTorch。"
        ) from exc

    extension = CUDAExtension(
        name="ampere_kv._C",
        sources=[
            str(PROJECT_ROOT / "csrc" / "bindings.cpp"),
            str(PROJECT_ROOT / "csrc" / "smoke_cuda.cu"),
        ],
        include_dirs=[str(PROJECT_ROOT / "csrc")],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            # v0.1 只为 RTX 3090/A10 的 Ampere sm_86 生成机器码。
            # 此处不启用 fast-math，避免 Smoke 阶段引入额外数值差异。
            "nvcc": [
                "-O3",
                "-std=c++17",
                "-gencode=arch=compute_86,code=sm_86",
                "--lineinfo",
            ],
        },
    )

    command_classes = {
        "build_ext": BuildExtension.with_options(no_python_abi_suffix=True),
    }
    return [extension], command_classes


ext_modules, cmdclass = create_cuda_extension()

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
