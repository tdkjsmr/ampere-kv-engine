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
        """复制外部算好的 K/V 到末尾；不记录梯度，不保留输入张量的引用。"""

        # 新数据必须是 [1, KV 头数, 本次 Token 数, 每头维度]，K/V 形状相同。
        # copy_ 本身允许广播和类型转换，这里主动拒绝，避免错误被悄悄掩盖。
        if key.ndim != 4 or key.shape != value.shape:
            raise ValueError("新 K/V 必须是形状相同的四维张量")
        if key.shape[0] != 1 or key.shape[1] != self._key.shape[1] or key.shape[3] != self._key.shape[3]:
            raise ValueError("新 K/V 的批大小、KV 头数或每头维度与缓存不一致")
        if key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16:
            raise ValueError("新 K/V 必须使用 BF16")
        if key.device != self._key.device or value.device != self._value.device:
            raise ValueError("新 K/V 必须与缓存位于同一设备")
        tokens = key.shape[2]
        if tokens == 0:
            raise ValueError("每次追加至少需要一个 Token")
        end = self.length + tokens
        if end > self.capacity:
            raise ValueError(f"缓存容量不足：已用 {self.length}，本次追加 {tokens}，容量 {self.capacity}")

        # 左闭右开区间 [length, end)：只覆盖空闲位置，不拼接或重新分配缓存。
        # 所有可预检错误都在复制前拒绝；这里不提供 GPU 故障下的事务回滚保证。
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
    """共用输入检查、缓存写入和 GQA 计算，避免两条路径复制同一套逻辑。"""

    # 先检查 Q 与头映射，避免非法输入已经写入缓存后才报错。
    if any(tensor.ndim != 4 for tensor in (query, key, value)):
        raise ValueError("Q/K/V 必须是四维张量")
    if any(tensor.shape[0] != 1 for tensor in (query, key, value)):
        raise ValueError("Attention 只支持单请求")
    if query.shape[2] == 0 or query.shape[2] != key.shape[2]:
        raise ValueError("Q/K 的本次 Token 数必须相同且非零")
    if is_causal:
        if cache.length != 0:
            raise ValueError("Prefill 只接受空缓存，不支持分块追加")
    elif query.shape[2] != 1:
        raise ValueError("Decode 只支持单 Token")
    if key.shape != value.shape or query.shape[-1] != key.shape[-1]:
        raise ValueError("K/V 形状必须相同，Q/K/V 每头维度必须一致")
    if query.shape[1] <= 0 or key.shape[1] <= 0 or query.shape[1] % key.shape[1] != 0:
        raise ValueError("Query 头数必须是 KV 头数的正整数倍")
    if any(tensor.dtype != torch.bfloat16 for tensor in (query, key, value)):
        raise ValueError("Q/K/V 必须使用 BF16")
    if query.device != key.device or key.device != value.device:
        raise ValueError("Q/K/V 必须位于同一设备")

    if cache._key.dtype not in (torch.bfloat16, torch.int8):
        raise ValueError("Attention 参考只支持 BF16 或 INT8 缓存")
    quantized = cache._key.dtype == torch.int8
    if quantized and not torch.isfinite(query).all().item():
        raise ValueError("INT8 Attention 参考的 Query 不能含 NaN 或 Inf")
    # append 继续负责检查 K/V 与缓存的形状、设备及容量是否匹配。
    cache.append(key, value)
    cached_key, cached_value = cache.get()
    # BF16 路径不变；INT8 get 已还原 FP32 历史，Query 也转 FP32，避免精度混用。
    # 这是独立反量化参考，不是融合 CUDA 路径，也不代表模型已支持 INT8。
    compute_query = query.float() if quantized else query
    compute_key, compute_value = cached_key, cached_value
    group_size = query.shape[1] // key.shape[1]
    if group_size > 1:
        # GQA 按 [KV0, KV0, KV1, KV1] 连续重复头；不改变缓存本体。
        # 这会复制临时数据，只是参考路径，不是高性能共享 KV 内核。
        compute_key = compute_key.repeat_interleave(group_size, dim=1)
        compute_value = compute_value.repeat_interleave(group_size, dim=1)
    # Prefill 从空缓存开始，Q/K 等长，用下三角掩码屏蔽后面的位置。
    # Decode 的 Q 只对应最后一个位置，有效 K/V 全部可见，不额外加三角掩码。
    return torch.nn.functional.scaled_dot_product_attention(
        # HF 4.51 的 SDPA 包装也先整理为连续布局；此处可能产生 BF16 临时副本。
        compute_query.contiguous(), compute_key.contiguous(), compute_value.contiguous(),
        dropout_p=0.0, is_causal=is_causal and query.shape[2] > 1,
        scale=query.shape[-1] ** -0.5,
    )
