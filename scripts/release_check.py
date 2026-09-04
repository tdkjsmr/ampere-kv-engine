"""检查 v0.1 G0/G1 证据是否齐全。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jsonschema import Draft202012Validator


def main() -> int:
    """校验环境指纹、HF Reference 和必需源码文件。"""

    parser = argparse.ArgumentParser(description="AmpereKV v0.1 Release Check")
    parser.add_argument("--run-dir", required=True, help="本次云端运行目录")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    run_dir = Path(args.run_dir)
    required_project_files = [
        "configs/model.lock.yaml",
        "configs/run_manifest.schema.json",
        "docs/model_contract.md",
        "docs/correctness_contract.md",
        "csrc/bindings.cpp",
        "csrc/smoke_cuda.cu",
        "triton_kernels/smoke_add.py",
    ]
    required_run_files = [
        run_dir / "environment.json",
        run_dir / "hf_reference.json",
        run_dir / "cuda_smoke.json",
        run_dir / "triton_smoke.json",
    ]

    errors = []
    for relative_path in required_project_files:
        if not (project_root / relative_path).is_file():
            errors.append(f"缺少项目文件：{relative_path}")
    for path in required_run_files:
        if not path.is_file():
            errors.append(f"缺少运行证据：{path}")

    environment_path = run_dir / "environment.json"
    if environment_path.is_file():
        schema = json.loads(
            (project_root / "configs/run_manifest.schema.json").read_text(
                encoding="utf-8"
            )
        )
        environment = json.loads(environment_path.read_text(encoding="utf-8"))
        for validation_error in Draft202012Validator(schema).iter_errors(
            environment
        ):
            errors.append(f"environment.json：{validation_error.message}")

    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1

    print("[PASS] v0.1 G0/G1 所需文件与运行证据齐全。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
