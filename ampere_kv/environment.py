"""生成不包含用户身份信息的 AmpereKV 环境指纹。"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ampere_kv.model.contract import ModelContract


def _package_version(name: str) -> str | None:
    """读取可选 Python 包版本；未安装时返回 None。"""

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _run_version_command(command: list[str]) -> str | None:
    """运行只读版本命令，并只保留第一条非空输出。

    环境指纹不保存完整命令输出，避免无意记录本地绝对路径、用户名或
    与复现无关的系统信息。
    """

    executable = shutil.which(command[0])
    if executable is None:
        return None

    completed = subprocess.run(
        [executable, *command[1:]],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    text = completed.stdout or completed.stderr
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return None


def _git_state(project_root: Path) -> dict[str, Any]:
    """获取 Git Commit 和已跟踪文件是否有修改。"""

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
    return {"commit": commit, "dirty": bool(status.strip())}


def _torch_hardware() -> dict[str, Any]:
    """通过 PyTorch 读取 GPU；没有 CUDA 时返回显式空值。"""

    try:
        import torch
    except ImportError:
        return {
            "gpu_name": None,
            "gpu_uuid": None,
            "compute_capability": None,
            "memory_total_bytes": None,
            "cuda_available": False,
        }

    if not torch.cuda.is_available():
        return {
            "gpu_name": None,
            "gpu_uuid": None,
            "compute_capability": None,
            "memory_total_bytes": None,
            "cuda_available": False,
        }

    properties = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability(0)
    return {
        "gpu_name": properties.name,
        # PyTorch 不保证所有版本公开 GPU UUID，因此该字段允许为空。
        "gpu_uuid": None,
        "compute_capability": f"{major}.{minor}",
        "memory_total_bytes": properties.total_memory,
        "cuda_available": True,
    }


def collect_environment(
    project_root: Path,
    run_id: str,
    lock_path: Path,
) -> dict[str, Any]:
    """收集符合 Run Manifest Schema 的基础环境信息。"""

    contract = ModelContract.from_lock_file(lock_path)
    hardware = _torch_hardware()

    try:
        import torch

        cuda_runtime = torch.version.cuda
    except ImportError:
        cuda_runtime = None

    return {
        "schema_version": 1,
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_state(project_root),
        "model": {
            "id": contract.model_id,
            "revision": contract.revision,
            "dtype": contract.torch_dtype,
        },
        "hardware": hardware,
        "software": {
            "python": platform.python_version(),
            "torch": _package_version("torch"),
            "triton": _package_version("triton"),
            "transformers": _package_version("transformers"),
            "cuda_runtime": cuda_runtime,
            "cuda_toolkit": _run_version_command(["nvcc", "--version"]),
            "driver": _run_version_command(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ]
            ),
            "ncu": _run_version_command(["ncu", "--version"]),
            "nsys": _run_version_command(["nsys", "--version"]),
            "gcc": _run_version_command(["gcc", "--version"]),
            "cmake": _run_version_command(["cmake", "--version"]),
            "platform": sys.platform,
        },
        "features": {
            "kv_dtype": "bfloat16",
            "cuda_graph": False,
            "chunked_prefill": False,
        },
    }


def main() -> int:
    """命令行入口：生成 environment.json。"""

    parser = argparse.ArgumentParser(description="生成 AmpereKV 环境指纹")
    parser.add_argument("--output", required=True, help="输出 JSON 路径")
    parser.add_argument("--run-id", default="dev", help="本次运行标识")
    parser.add_argument(
        "--lock",
        default="configs/model.lock.yaml",
        help="模型锁文件路径",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    manifest = collect_environment(
        project_root=project_root,
        run_id=args.run_id,
        lock_path=project_root / args.lock,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[PASS] 环境指纹已写入：{output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
