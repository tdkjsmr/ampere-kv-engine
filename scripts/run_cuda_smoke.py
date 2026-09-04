"""运行 AmpereKV 自定义 CUDA Extension Smoke。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ampere_kv.smoke import run_cuda_smoke


def main() -> int:
    """运行 Smoke，并以 JSON 打印可保存结果。"""

    parser = argparse.ArgumentParser(description="运行 CUDA Extension Smoke")
    parser.add_argument(
        "--output",
        help="可选的 JSON 输出路径；未指定时只输出到终端",
    )
    args = parser.parse_args()

    result = run_cuda_smoke()
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    print(serialized)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")
        print(f"[PASS] CUDA Smoke 证据已写入：{output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
