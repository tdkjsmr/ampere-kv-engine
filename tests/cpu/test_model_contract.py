"""Qwen3-8B 模型锁与配置契约测试。"""

from __future__ import annotations

import json
from pathlib import Path

from ampere_kv.model.contract import ModelContract, validate_model_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_official_fixture() -> dict:
    """读取固定 Revision 对应的官方 config.json Fixture。"""

    path = PROJECT_ROOT / "tests" / "fixtures" / "qwen3_8b_config.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_official_config_matches_lock() -> None:
    """固定官方配置必须与 model.lock.yaml 完全一致。"""

    contract = ModelContract.from_lock_file(
        PROJECT_ROOT / "configs" / "model.lock.yaml"
    )
    assert validate_model_config(contract, _load_official_fixture()) == []
    assert contract.shape.gqa_group_size == 4


def test_contract_reports_all_changed_fields() -> None:
    """一次报告所有漂移字段，减少云端反复修改和重跑。"""

    contract = ModelContract.from_lock_file(
        PROJECT_ROOT / "configs" / "model.lock.yaml"
    )
    changed = _load_official_fixture()
    changed["num_hidden_layers"] = 35
    changed["head_dim"] = 256

    errors = validate_model_config(contract, changed)
    assert any("num_hidden_layers" in error for error in errors)
    assert any("head_dim" in error for error in errors)
