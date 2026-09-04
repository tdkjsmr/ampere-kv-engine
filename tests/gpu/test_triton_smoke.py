"""最小 Triton Kernel 的云端 GPU Smoke 测试。"""

from __future__ import annotations

import pytest


@pytest.mark.gpu
def test_triton_smoke_with_masked_tail() -> None:
    """非整 Block 长度必须通过 Mask 正确处理尾部元素。"""

    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("当前测试环境没有 CUDA")

    from triton_kernels import triton_smoke_add

    left = torch.arange(1025, device="cuda", dtype=torch.float32)
    right = torch.ones_like(left)
    actual = triton_smoke_add(left, right)

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, left + right, rtol=0.0, atol=0.0)
