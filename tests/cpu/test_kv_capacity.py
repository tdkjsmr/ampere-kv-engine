"""KV Cache 理论容量账本的 CPU 单元测试。"""

from __future__ import annotations

import math

import pytest

from ampere_kv.cache.kv_capacity import ModelShape, calculate_kv_capacity

QWEN3_8B_SHAPE = ModelShape(
    num_hidden_layers=36,
    num_attention_heads=32,
    num_key_value_heads=8,
    head_dim=128,
)


def test_qwen3_8b_gqa_group_size() -> None:
    """Qwen3-8B 每个 KV Head 必须对应四个 Query Head。"""

    assert QWEN3_8B_SHAPE.gqa_group_size == 4


def test_qwen3_8b_bytes_per_token() -> None:
    """校验 BF16 数据、INT8 数据和 Scale 的逐 Token 公式。"""

    report = calculate_kv_capacity(QWEN3_8B_SHAPE, context_tokens=1)

    assert report.bf16_data_bytes_per_token == 147_456
    assert report.int8_data_bytes_per_token == 73_728
    assert report.int8_scale_bytes_per_token == 1_152
    assert report.int8_total_bytes_per_token == 74_880
    assert math.isclose(
        report.theoretical_capacity_ratio,
        147_456 / 74_880,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_qwen3_8b_32768_context_total() -> None:
    """校验计划中 32768 Token 工作负载的理论 KV 总量。"""

    report = calculate_kv_capacity(QWEN3_8B_SHAPE, context_tokens=32_768)

    assert report.bf16_total_bytes == 4_831_838_208
    assert report.int8_total_bytes == 2_453_667_840


def test_invalid_gqa_shape_is_rejected() -> None:
    """Query Head 无法整除 KV Head 时必须拒绝，而不是静默截断。"""

    invalid = ModelShape(
        num_hidden_layers=36,
        num_attention_heads=30,
        num_key_value_heads=8,
        head_dim=128,
    )
    with pytest.raises(ValueError, match="整除"):
        invalid.validate()


def test_negative_context_is_rejected() -> None:
    """负 Context 长度没有物理意义，必须显式报错。"""

    with pytest.raises(ValueError, match="不能为负数"):
        calculate_kv_capacity(QWEN3_8B_SHAPE, context_tokens=-1)
