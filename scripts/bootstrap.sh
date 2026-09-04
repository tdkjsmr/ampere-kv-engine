#!/usr/bin/env bash

# AmpereKV v0.1 云端开发环境初始化脚本。
# 本脚本面向 Ubuntu 22.04 + CUDA 12.4 + RTX 3090，不在本地 Windows 执行。

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "[FAIL] bootstrap.sh 只支持 Linux 云端环境。" >&2
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[FAIL] 未找到 Python：${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# PyTorch CUDA Wheel 必须从官方 CUDA 12.4 索引安装。
python -m pip install \
  torch==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124

python -m pip install \
  transformers==4.51.0 \
  "huggingface-hub>=0.24"

python -m pip install -e ".[dev]"

echo "[PASS] AmpereKV v0.1 开发环境已准备完成。"
echo "下一步：source ${VENV_DIR}/bin/activate && make verify-env"
