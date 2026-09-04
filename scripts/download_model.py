"""按 model.lock.yaml 下载并校验 Qwen3-8B 文件。"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from ampere_kv.config import load_yaml_mapping, require_mapping
from ampere_kv.model.contract import sha256_file

METADATA_FILES = [
    "config.json",
    "tokenizer_config.json",
    "generation_config.json",
]


def main() -> int:
    """默认只下载元数据；显式允许时才下载模型权重。"""

    parser = argparse.ArgumentParser(description="下载锁定版本的 Qwen3-8B")
    parser.add_argument(
        "--lock",
        default="configs/model.lock.yaml",
        help="模型锁文件",
    )
    parser.add_argument(
        "--local-dir",
        default="models/Qwen3-8B",
        help="本地模型目录；该目录默认被 Git 忽略",
    )
    parser.add_argument(
        "--include-weights",
        action="store_true",
        help="同时下载 Safetensors 权重；默认只下载元数据",
    )
    args = parser.parse_args()

    lock = load_yaml_mapping(args.lock)
    model = require_mapping(lock, "model")
    metadata = require_mapping(lock, "metadata_files")

    allow_patterns = None
    if not args.include_weights:
        allow_patterns = [
            *METADATA_FILES,
            "tokenizer.json",
            "vocab.json",
            "merges.txt",
            "chat_template.jinja",
        ]

    local_dir = Path(args.local_dir)
    snapshot_download(
        repo_id=model["id"],
        revision=model["revision"],
        local_dir=local_dir,
        allow_patterns=allow_patterns,
    )

    errors = []
    for filename in METADATA_FILES:
        expected_hash = metadata[filename]["sha256"]
        actual_hash = sha256_file(local_dir / filename)
        if actual_hash != expected_hash:
            errors.append(
                f"{filename}: 期望 {expected_hash}，实际 {actual_hash}"
            )

    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1

    mode = "元数据和权重" if args.include_weights else "元数据"
    print(f"[PASS] 已下载并校验锁定版本的 Qwen3-8B {mode}。")
    print(f"[INFO] Revision：{model['revision']}")
    print(f"[INFO] 本地目录：{local_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
