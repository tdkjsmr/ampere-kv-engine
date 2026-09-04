"""最小 CUDA Extension Smoke 的 Python 调用层。"""

from __future__ import annotations


def run_cuda_smoke() -> dict[str, object]:
    """运行自定义 CUDA 加法算子并与 PyTorch Reference 比较。"""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，必须在云端 GPU 环境运行 Smoke")

    # 导入扩展时会完成 torch.library 算子注册。
    from ampere_kv import _C  # noqa: F401

    left = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
    right = torch.tensor([10.0, 20.0, 30.0, 40.0], device="cuda")
    actual = torch.ops.ampere_kv.smoke_add(left, right)
    expected = left + right

    # 显式同步，确保异步 Kernel 启动错误不会在进程退出后被遗漏。
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    major, minor = torch.cuda.get_device_capability(0)
    return {
        "status": "PASS",
        "gpu_name": torch.cuda.get_device_name(0),
        "compute_capability": f"{major}.{minor}",
        "dtype": str(actual.dtype),
        "result": actual.cpu().tolist(),
    }
