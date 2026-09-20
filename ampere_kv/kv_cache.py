"""单层 BF16 连续 KV 存储与共用 Prefill/Decode Attention。"""

import torch


class ContiguousKVCache:
    """固定容量、只允许尾部追加的推理缓存，不支持扩容或任意位置覆盖。"""

    def __init__(self, num_kv_heads: int, head_dim: int, capacity: int, device="cpu"):
        # 当前批大小固定为 1；头数、每头维度和容量由调用方明确给出。
        if num_kv_heads <= 0 or head_dim <= 0 or capacity <= 0:
            raise ValueError("KV 头数、每头维度和容量必须为正数")
        self.capacity = capacity
        self.length = 0
        shape = (1, num_kv_heads, capacity, head_dim)
        # empty 不初始化数据：只有已经写入的前缀才有效，未写入区域绝不能使用。
        self._key = torch.empty(shape, dtype=torch.bfloat16, device=device)
        self._value = torch.empty(shape, dtype=torch.bfloat16, device=device)

    @torch.no_grad()
    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """复制外部算好的 K/V 到末尾 [length, length+tokens)；不拼接或重新分配缓存。"""
        tokens = key.shape[2]
        end = self.length + tokens
        if end > self.capacity:
            raise ValueError(f"缓存容量不足：已用 {self.length}，本次追加 {tokens}，容量 {self.capacity}")
        self._key[:, :, self.length:end, :].copy_(key)
        self._value[:, :, self.length:end, :].copy_(value)
        self.length = end

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        """返回有效前缀视图；调用方应只读，修改返回值也会修改缓存。"""

        # 不返回未初始化的尾部，不额外复制历史数据。
        # 多头情况下，容量大于有效长度时，这个视图可能不是 contiguous 张量。
        return self._key[:, :, :self.length, :], self._value[:, :, :self.length, :]


def prefill_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, cache,
) -> torch.Tensor:
    """向空缓存写入整段 BF16 K/V，返回全部输入位置的因果 Attention 输出。

    BF16 缓存输出 BF16；INT8 分页缓存反量化后用 FP32 计算并输出，仅作参考。
    不支持分块 Prefill 或填充输入；追加后若计算失败，不自动回滚缓存。
    """
    return _cached_attention(query, key, value, cache, is_causal=True)


def decode_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, cache,
) -> torch.Tensor:
    """追加当前单 Token 的 BF16 K/V，返回 [1, Query 头数, 1, 每头维度]。

    BF16 缓存输出 BF16；INT8 分页缓存的反量化参考输出 FP32。

    接受提供 length、append、get 的连续或分页缓存；无需继承公共基类。
    调用前必须已有历史，调用方不要提前追加当前 K/V；重复调用会重复追加。
    输入检查在写入前完成；追加后若计算失败，不自动回滚缓存，不能盲目重试。
    这是推理参考计算，不包含 QKV 投影、位置编码、头合并或输出投影。
    """
    if cache.length == 0:
        raise ValueError("Decode 前必须已有历史 K/V")
    return _cached_attention(query, key, value, cache, is_causal=False)


@torch.no_grad()
def _cached_attention(query, key, value, cache, *, is_causal: bool) -> torch.Tensor:
    """共用缓存写入、GQA 展开与 SDPA；Q/K/V 由调用方的模型前向产出，这里不再重复校验。"""
    cache.append(key, value)
    cached_key, cached_value = cache.get()
    # INT8 缓存的 get 已还原 FP32 历史，Query 同步转 FP32，避免精度混用。
    compute_query = query.float() if cache._key.dtype == torch.int8 else query
    compute_key, compute_value = cached_key, cached_value
    group_size = query.shape[1] // key.shape[1]
    if group_size > 1:
        # GQA 按 [KV0, KV0, KV1, KV1] 连续重复头；只是参考路径的临时副本，不改变缓存本体。
        compute_key = compute_key.repeat_interleave(group_size, dim=1)
        compute_value = compute_value.repeat_interleave(group_size, dim=1)
    # Prefill 从空缓存开始，Q/K 等长，用下三角掩码；Decode 的 Q 只对应最后一个位置。
    return torch.nn.functional.scaled_dot_product_attention(
        compute_query.contiguous(), compute_key.contiguous(), compute_value.contiguous(),
        dropout_p=0.0, is_causal=is_causal and query.shape[2] > 1,
        scale=query.shape[-1] ** -0.5,
    )
