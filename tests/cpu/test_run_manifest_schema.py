"""Run Manifest JSON Schema 的 CPU 单元测试。"""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _schema() -> dict:
    """读取 v0.1 Run Manifest Schema。"""

    return json.loads(
        (PROJECT_ROOT / "configs" / "run_manifest.schema.json").read_text(
            encoding="utf-8"
        )
    )


def _valid_manifest() -> dict:
    """构造不依赖真实 GPU 的最小合法 Manifest。"""

    return {
        "schema_version": 1,
        "run_id": "cpu-schema-test",
        "created_at_utc": "2026-09-04T00:00:00+00:00",
        "git": {
            "commit": "0" * 40,
            "dirty": False,
        },
        "model": {
            "id": "Qwen/Qwen3-8B",
            "revision": "b968826d9c46dd6066d109eabc6255188de91218",
            "dtype": "bfloat16",
        },
        "hardware": {
            "gpu_name": None,
            "gpu_uuid": None,
            "compute_capability": None,
            "memory_total_bytes": None,
        },
        "software": {
            "python": "3.11.0",
            "torch": None,
            "triton": None,
            "transformers": None,
            "cuda_runtime": None,
            "cuda_toolkit": None,
            "driver": None,
        },
        "features": {
            "kv_dtype": "bfloat16",
            "cuda_graph": False,
            "chunked_prefill": False,
        },
    }


def test_valid_manifest_passes() -> None:
    """合法 Manifest 不应产生 Schema 错误。"""

    errors = list(Draft202012Validator(_schema()).iter_errors(_valid_manifest()))
    assert errors == []


def test_floating_git_revision_is_rejected() -> None:
    """正式运行不能使用 main/latest 等浮动模型或代码标识。"""

    manifest = _valid_manifest()
    manifest["git"]["commit"] = "main"
    errors = list(Draft202012Validator(_schema()).iter_errors(manifest))
    assert any("does not match" in error.message for error in errors)


def test_unknown_feature_is_rejected() -> None:
    """未知功能开关必须先更新 Schema，不能静默混入结果。"""

    manifest = _valid_manifest()
    manifest["features"]["unknown_optimization"] = True
    errors = list(Draft202012Validator(_schema()).iter_errors(manifest))
    assert any(error.validator == "additionalProperties" for error in errors)
