#!/usr/bin/env bash

# 生成环境指纹，并对 G0 所需的 RTX 3090 / sm_86 条件做快速校验。

set -euo pipefail

OUTPUT=""
RUN_ID="dev"
ALLOW_NO_CUDA="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --allow-no-cuda)
      ALLOW_NO_CUDA="true"
      shift
      ;;
    *)
      echo "[FAIL] 未知参数：$1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${OUTPUT}" ]]; then
  echo "[FAIL] 必须通过 --output 指定 environment.json 路径。" >&2
  exit 2
fi

for command_name in git python nvcc nvidia-smi cmake; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "[FAIL] 缺少必需命令：${command_name}" >&2
    exit 1
  fi
done

python -m ampere_kv.environment \
  --output "${OUTPUT}" \
  --run-id "${RUN_ID}"

python - "${OUTPUT}" "${ALLOW_NO_CUDA}" <<'PY'
import json
import sys
from pathlib import Path

# 这里只读取刚生成的 JSON，不再次查询硬件，保证检查对象和保存对象一致。
manifest_path = Path(sys.argv[1])
allow_no_cuda = sys.argv[2] == "true"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
hardware = manifest["hardware"]

if not hardware.get("cuda_available", False):
    if allow_no_cuda:
        print("[WARN] 当前环境没有 CUDA；仅完成非 GPU 环境指纹采集。")
        raise SystemExit(0)
    raise SystemExit("[FAIL] PyTorch 无法使用 CUDA。")

if hardware.get("compute_capability") != "8.6":
    raise SystemExit(
        "[FAIL] 目标 Compute Capability 必须为 8.6，"
        f"实际为 {hardware.get('compute_capability')!r}。"
    )

print(f"[PASS] GPU：{hardware.get('gpu_name')}")
print("[PASS] Compute Capability：8.6（sm_86）")
print(f"[PASS] 环境指纹：{manifest_path}")
PY
