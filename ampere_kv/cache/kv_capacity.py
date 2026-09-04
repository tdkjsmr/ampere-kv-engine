"""Qwen3 GQA KV Cache 的可解释容量账本。

本模块只做纯整数计算，不依赖 PyTorch 或 CUDA。这样可以在本地 Windows、
CPU CI 和云端环境中得到完全一致的理论值。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

BYTES_PER_BF16 = 2
BYTES_PER_INT8 = 1
BYTES_PER_FP16_SCALE = 2


@dataclass(frozen=True)
class ModelShape:
    """计算 KV Cache 所需的最小模型形状。"""

    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int

    def validate(self) -> None:
        """检查形状合法性和 GQA 整除关系。"""

        fields = {
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
        }
        for name, value in fields.items():
            if value <= 0:
                raise ValueError(f"{name} 必须大于 0，实际为 {value}")

        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "Query Head 数必须能被 KV Head 数整除，才能形成固定 GQA 分组"
            )

    @property
    def gqa_group_size(self) -> int:
        """返回每个 KV Head 对应的 Query Head 数。"""

        self.validate()
        return self.num_attention_heads // self.num_key_value_heads


@dataclass(frozen=True)
class KVCapacityReport:
    """单 Token 与指定上下文长度的 KV 理论字节账本。"""

    context_tokens: int
    gqa_group_size: int
    bf16_data_bytes_per_token: int
    int8_data_bytes_per_token: int
    int8_scale_bytes_per_token: int
    int8_total_bytes_per_token: int
    bf16_total_bytes: int
    int8_total_bytes: int

    @property
    def theoretical_capacity_ratio(self) -> float:
        """返回相同字节预算下 INT8 KV 相对 BF16 KV 的理论容量倍数。"""

        return self.bf16_data_bytes_per_token / self.int8_total_bytes_per_token

    def to_dict(self) -> dict[str, int | float]:
        """转换为适合 JSON 序列化的字典。"""

        result = asdict(self)
        result["theoretical_capacity_ratio"] = self.theoretical_capacity_ratio
        return result


def calculate_kv_capacity(shape: ModelShape, context_tokens: int) -> KVCapacityReport:
    """计算 BF16 KV 与 Per-token/head INT8 KV 的理论占用。

    KV 数据布局同时保存 K 和 V，所以公式最前面有系数 2。INT8 Scale
    也分别为 K 和 V 保存；每个 Token、每层、每个 KV Head 各有一个
    FP16 Scale。该函数只计算 KV 与 Scale，不包含模型权重、激活、
    CUDA Graph Static Buffer、Block Table 或分配器碎片。
    """

    shape.validate()
    if context_tokens < 0:
        raise ValueError("context_tokens 不能为负数")

    kv_elements_per_token = (
        2
        * shape.num_hidden_layers
        * shape.num_key_value_heads
        * shape.head_dim
    )
    bf16_data_bytes_per_token = kv_elements_per_token * BYTES_PER_BF16
    int8_data_bytes_per_token = kv_elements_per_token * BYTES_PER_INT8

    int8_scale_bytes_per_token = (
        2
        * shape.num_hidden_layers
        * shape.num_key_value_heads
        * BYTES_PER_FP16_SCALE
    )
    int8_total_bytes_per_token = (
        int8_data_bytes_per_token + int8_scale_bytes_per_token
    )

    return KVCapacityReport(
        context_tokens=context_tokens,
        gqa_group_size=shape.gqa_group_size,
        bf16_data_bytes_per_token=bf16_data_bytes_per_token,
        int8_data_bytes_per_token=int8_data_bytes_per_token,
        int8_scale_bytes_per_token=int8_scale_bytes_per_token,
        int8_total_bytes_per_token=int8_total_bytes_per_token,
        bf16_total_bytes=bf16_data_bytes_per_token * context_tokens,
        int8_total_bytes=int8_total_bytes_per_token * context_tokens,
    )
