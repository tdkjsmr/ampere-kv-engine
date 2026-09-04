"""自定义 CUDA Extension 的云端 GPU Smoke 测试。"""

from __future__ import annotations

import pytest


@pytest.mark.gpu
def test_cuda_extension_smoke() -> None:
    """Extension 必须完成真实 Kernel 执行与 Reference 对齐。"""

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("当前测试环境没有 CUDA")

    from ampere_kv.smoke import run_cuda_smoke

    result = run_cuda_smoke()
    assert result["status"] == "PASS"
    assert result["compute_capability"] == "8.6"
    assert result["result"] == [11.0, 22.0, 33.0, 44.0]
