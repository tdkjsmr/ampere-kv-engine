"""运行 AmpereKV 最小 Triton Kernel Smoke。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from triton_kernels import triton_smoke_add


def main() -> int:
    """使用非整 Block 长度验证 Mask 和 Kernel 启动链路。"""

    parser = argparse.ArgumentParser(description="运行 Triton Smoke")
    parser.add_argument(
        "--output",
        help="可选的 JSON 输出路径；未指定时只输出到终端",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，必须在云端 GPU 环境运行 Triton Smoke")

    # 1025 故意不能被 256 整除，用于覆盖最后一个 Program 的 Mask。
    left = torch.arange(1025, device="cuda", dtype=torch.float32)
    right = torch.full_like(left, 3.0)
    actual = triton_smoke_add(left, right)
    expected = left + right

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    major, minor = torch.cuda.get_device_capability(0)

    result = {
        "status": "PASS",
        "gpu_name": torch.cuda.get_device_name(0),
        "compute_capability": f"{major}.{minor}",
        "num_elements": actual.numel(),
        "tail_value": actual[-1].item(),
    }
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    print(serialized)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")
        print(f"[PASS] Triton Smoke 证据已写入：{output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
