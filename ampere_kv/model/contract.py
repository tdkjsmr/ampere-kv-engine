"""锁定并校验 Qwen3-8B 的模型配置契约。"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ampere_kv.cache.kv_capacity import ModelShape, calculate_kv_capacity
from ampere_kv.config import (
    ConfigError,
    load_yaml_mapping,
    require_mapping,
    require_value,
)


@dataclass(frozen=True)
class ModelContract:
    """v0.1 后所有模型代码必须遵守的固定 Qwen3 契约。"""

    model_id: str
    revision: str
    tokenizer_revision: str
    architecture: str
    model_type: str
    parameter_count: int
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    vocab_size: int
    rope_theta: int
    tie_word_embeddings: bool
    torch_dtype: str

    @classmethod
    def from_lock_file(cls, path: str | Path) -> ModelContract:
        """从 model.lock.yaml 创建契约对象。"""

        root = load_yaml_mapping(path)
        model = require_mapping(root, "model")
        contract = require_mapping(root, "contract")

        return cls(
            model_id=require_value(model, "id", str),
            revision=require_value(model, "revision", str),
            tokenizer_revision=require_value(model, "tokenizer_revision", str),
            architecture=require_value(contract, "architecture", str),
            model_type=require_value(contract, "model_type", str),
            parameter_count=require_value(contract, "parameter_count", int),
            num_hidden_layers=require_value(contract, "num_hidden_layers", int),
            hidden_size=require_value(contract, "hidden_size", int),
            intermediate_size=require_value(contract, "intermediate_size", int),
            num_attention_heads=require_value(contract, "num_attention_heads", int),
            num_key_value_heads=require_value(contract, "num_key_value_heads", int),
            head_dim=require_value(contract, "head_dim", int),
            max_position_embeddings=require_value(
                contract, "max_position_embeddings", int
            ),
            vocab_size=require_value(contract, "vocab_size", int),
            rope_theta=require_value(contract, "rope_theta", int),
            tie_word_embeddings=require_value(
                contract, "tie_word_embeddings", bool
            ),
            torch_dtype=require_value(contract, "torch_dtype", str),
        )

    @property
    def shape(self) -> ModelShape:
        """转换为 KV 容量计算所需的形状。"""

        return ModelShape(
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
        )


def _actual_contract_fields(config: dict[str, Any]) -> dict[str, Any]:
    """从 Hugging Face config.json 中提取我们冻结的字段。"""

    architectures = config.get("architectures")
    if not isinstance(architectures, list) or len(architectures) != 1:
        raise ConfigError("architectures 必须是只包含一个模型类名的列表")

    return {
        "architecture": architectures[0],
        "model_type": config.get("model_type"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "hidden_size": config.get("hidden_size"),
        "intermediate_size": config.get("intermediate_size"),
        "num_attention_heads": config.get("num_attention_heads"),
        "num_key_value_heads": config.get("num_key_value_heads"),
        "head_dim": config.get("head_dim"),
        "max_position_embeddings": config.get("max_position_embeddings"),
        "vocab_size": config.get("vocab_size"),
        "rope_theta": config.get("rope_theta"),
        "tie_word_embeddings": config.get("tie_word_embeddings"),
        "torch_dtype": config.get("torch_dtype"),
    }


def validate_model_config(
    contract: ModelContract, config: dict[str, Any]
) -> list[str]:
    """逐字段比较官方配置与锁文件，返回全部错误而不是遇到首错即退出。"""

    expected = {
        "architecture": contract.architecture,
        "model_type": contract.model_type,
        "num_hidden_layers": contract.num_hidden_layers,
        "hidden_size": contract.hidden_size,
        "intermediate_size": contract.intermediate_size,
        "num_attention_heads": contract.num_attention_heads,
        "num_key_value_heads": contract.num_key_value_heads,
        "head_dim": contract.head_dim,
        "max_position_embeddings": contract.max_position_embeddings,
        "vocab_size": contract.vocab_size,
        "rope_theta": contract.rope_theta,
        "tie_word_embeddings": contract.tie_word_embeddings,
        "torch_dtype": contract.torch_dtype,
    }
    actual = _actual_contract_fields(config)

    errors = []
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if actual_value != expected_value:
            errors.append(
                f"{key}: 期望 {expected_value!r}，实际 {actual_value!r}"
            )

    try:
        contract.shape.validate()
    except ValueError as exc:
        errors.append(f"GQA 形状非法：{exc}")
    return errors


def sha256_file(path: str | Path) -> str:
    """流式计算文件 SHA-256，避免把大型文件一次性读入内存。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    """命令行校验入口。"""

    parser = argparse.ArgumentParser(description="校验 Qwen3-8B 模型契约")
    parser.add_argument("--lock", required=True, help="model.lock.yaml 路径")
    parser.add_argument("--config", required=True, help="config.json 路径")
    args = parser.parse_args()

    contract = ModelContract.from_lock_file(args.lock)
    with Path(args.config).open("r", encoding="utf-8") as file:
        config = json.load(file)

    errors = validate_model_config(contract, config)
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1

    capacity = calculate_kv_capacity(contract.shape, context_tokens=32768)
    print("[PASS] Qwen3-8B 模型契约与锁文件一致")
    print(json.dumps(capacity.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
