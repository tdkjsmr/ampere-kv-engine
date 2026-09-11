"""单层 BF16 连续 KV 存储，附 Prefill、Decode 与 GQA 的 CPU 因果 Attention 对照。"""

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


def main() -> None:
    """显式运行本模块时，在 CPU 上检查存储和 Attention 对照；不使用 GPU。"""

    cache = ContiguousKVCache(num_kv_heads=2, head_dim=4, capacity=6)
    # 地址只用于验证原地追加，没有在类的公开接口中增加调试功能。
    addresses = (cache._key.data_ptr(), cache._value.data_ptr())
    assert cache.length == 0
    assert all(tensor.shape == (1, 2, 0, 4) for tensor in cache.get())

    # 用小整数构造容易核对的 BF16 数据；K/V 不相同，便于发现混写错误。
    keys = torch.arange(48, dtype=torch.float32).reshape(1, 2, 6, 4).to(torch.bfloat16)
    values = keys + 64
    start = 0
    for tokens in (3, 1, 2):
        # 依次覆盖：多 Token 写入、单 Token 追加、恰好写满容量。
        end = start + tokens
        cache.append(keys[:, :, start:end, :], values[:, :, start:end, :])
        actual_key, actual_value = cache.get()
        assert cache.length == end
        # 对照整个有效前缀：既检查新数据，也检查历史数据没有被覆盖。
        assert torch.equal(actual_key, keys[:, :, :end, :])
        assert torch.equal(actual_value, values[:, :, :end, :])
        assert (cache._key.data_ptr(), cache._value.data_ptr()) == addresses
        print(f"[PASS] 追加 {tokens} 个 Token：有效长度={end}，内容正确，存储地址不变")
        start = end

    # 容量已满，再写一个 Token 必须失败；失败后长度、数据和地址均不变。
    try:
        cache.append(keys[:, :, :1, :], values[:, :, :1, :])
    except ValueError as error:
        print(f"[PASS] 越界写入被拒绝：{error}")
    else:
        raise AssertionError("[FAIL] 缓存已满却仍允许追加")
    actual_key, actual_value = cache.get()
    assert cache.length == 6
    assert torch.equal(actual_key, keys) and torch.equal(actual_value, values)
    assert (cache._key.data_ptr(), cache._value.data_ptr()) == addresses
    print("[PASS] 单层连续 KV 存储自检通过；此结果不代表模型或 GPU 验证通过")

    # 新场景：前 3 个位置做 Prefill，第 4 个位置做 Decode；两个头不使用 GQA。
    # 使用局部随机数生成器固定输入，不改变外部程序的全局随机状态。
    generator = torch.Generator().manual_seed(0)
    query = torch.randn(1, 2, 1, 4, generator=generator).to(torch.bfloat16)
    full_key = torch.randn(1, 2, 4, 4, generator=generator).to(torch.bfloat16)
    full_value = torch.randn(1, 2, 4, 4, generator=generator).to(torch.bfloat16)
    prefix_query = torch.randn(1, 2, 3, 4, generator=generator).to(torch.bfloat16)
    full_query = torch.cat([prefix_query, query], dim=2)

    # 独立参考一次计算全部 4 个位置，显式屏蔽未来；不调用我们的两个接口。
    future_mask = torch.ones(4, 4, dtype=torch.bool).triu(diagonal=1)
    # 将缓存接入与独立数学对照分开；本函数仅供当前 CPU 小张量自检使用。
    def check_attention(label, q, k, v, actual, formula):
        # 用显式索引建立 GQA 头映射，不复用被测路径的 repeat_interleave。
        # 等头数映射为 [0, 1]，4 个 Query 头映射为 [0, 0, 1, 1]。
        mapping = torch.arange(q.shape[1]) // (q.shape[1] // k.shape[1])
        direct_key = k.index_select(1, mapping)
        direct_value = v.index_select(1, mapping)
        for dtype in (torch.bfloat16, torch.float32):
            # 两种精度使用相同原始输入；BF16 对照严格匹配被测路径的分段形状。
            direct_query = q.to(dtype)
            key_input, value_input = direct_key.to(dtype), direct_value.to(dtype)
            prefix = torch.nn.functional.scaled_dot_product_attention(
                direct_query[:, :, :3].contiguous(), key_input[:, :, :3].contiguous(),
                value_input[:, :, :3].contiguous(), dropout_p=0.0, is_causal=True,
                scale=q.shape[-1] ** -0.5,
            )
            # 最后一个 Query 可以看见全部 4 个有效位置；这里不读取我们的缓存。
            last = torch.nn.functional.scaled_dot_product_attention(
                direct_query[:, :, 3:].contiguous(), key_input.contiguous(), value_input.contiguous(),
                dropout_p=0.0, is_causal=False, scale=q.shape[-1] ** -0.5,
            )
            direct = torch.cat([prefix, last], dim=2)
            if dtype == torch.bfloat16:
                # 相同精度和执行形状下要求完全一致，检查缓存接入与 GQA 映射。
                torch.testing.assert_close(actual, direct, rtol=0, atol=0)
                print(f"[PASS] {label}：缓存与无缓存 BF16 分段输出完全一致")
            else:
                # 独立公式包含显式因果掩码；FP32 阈值仅用于当前小张量，非模型或量化标准。
                torch.testing.assert_close(direct, formula, rtol=1e-5, atol=1e-6)
                error = (direct - formula).abs().max().item()
                print(f"[PASS] {label}：FP32 分段 SDPA 与整段因果公式对照，最大绝对误差={error:.8g}")
        # 跨精度只做观测，不伪装成精度验收；NaN/Inf 仍必须拒绝。
        assert torch.isfinite(actual).all() and torch.isfinite(formula).all()
        error = (actual.float() - formula).abs().max().item()
        print(f"[观测] {label}：BF16 / FP32 公式最大绝对误差={error:.8g}；不代表跨精度验收通过")

    scores = full_query.float() @ full_key.float().transpose(-2, -1)
    scores = (scores / query.shape[-1] ** 0.5).masked_fill(future_mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    expected_output = weights @ full_value.float()

    # 容量故意大于有效长度，检查 get() 不会把未初始化的空闲位置交给 Attention。
    attention_cache = ContiguousKVCache(num_kv_heads=2, head_dim=4, capacity=6)
    prefill_output = prefill_attention(prefix_query, full_key[:, :, :3], full_value[:, :, :3], attention_cache)
    assert attention_cache.length == 3
    assert prefill_output.dtype == torch.bfloat16
    # 两个接口各自负责写入自己的 K/V，自检不重复追加。
    actual_output = decode_attention(query, full_key[:, :, 3:4, :], full_value[:, :, 3:4, :], attention_cache)
    assert actual_output.dtype == torch.bfloat16
    actual_output = torch.cat([prefill_output, actual_output], dim=2)
    cached_key, cached_value = attention_cache.get()
    assert attention_cache.length == 4
    assert cached_key.shape == cached_value.shape == (1, 2, 4, 4)
    assert torch.equal(cached_key, full_key) and torch.equal(cached_value, full_value)

    check_attention("等头数 Prefill + Decode", full_query, full_key, full_value, actual_output, expected_output)

    # GQA：沿用原始 K/V 数据，但另建缓存；4 个 Query 头共享 2 个 KV 头。
    gqa_query = torch.randn(1, 4, 1, 4, generator=generator).to(torch.bfloat16)
    gqa_prefix = torch.randn(1, 4, 3, 4, generator=generator).to(torch.bfloat16)
    full_gqa_query = torch.cat([gqa_prefix, gqa_query], dim=2)
    assert gqa_query.shape[1] % cached_key.shape[1] == 0
    group_size = gqa_query.shape[1] // cached_key.shape[1]
    # 独立参考不读缓存、不扩展 K/V：逐个 Query 头按整数除法找到对应的原始 KV 头。
    head_outputs = []
    for query_head in range(gqa_query.shape[1]):
        kv_head = query_head // group_size  # 0、1 映射到 0；2、3 映射到 1。
        head_query = full_gqa_query[:, query_head:query_head + 1].float()
        head_key = full_key[:, kv_head:kv_head + 1].float()
        head_value = full_value[:, kv_head:kv_head + 1].float()
        scores = head_query @ head_key.transpose(-2, -1)
        scores = (scores / gqa_query.shape[-1] ** 0.5).masked_fill(future_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        head_outputs.append(weights @ head_value)
    expected_gqa = torch.cat(head_outputs, dim=1)

    gqa_cache = ContiguousKVCache(num_kv_heads=2, head_dim=4, capacity=6)
    gqa_prefill = prefill_attention(gqa_prefix, full_key[:, :, :3], full_value[:, :, :3], gqa_cache)
    assert gqa_cache.length == 3
    assert gqa_prefill.dtype == torch.bfloat16
    actual_gqa = decode_attention(gqa_query, full_key[:, :, 3:4, :], full_value[:, :, 3:4, :], gqa_cache)
    assert actual_gqa.dtype == torch.bfloat16
    actual_gqa = torch.cat([gqa_prefill, actual_gqa], dim=2)
    assert gqa_cache.length == 4
    assert gqa_cache.get()[0].shape == gqa_cache.get()[1].shape == (1, 2, 4, 4)
    assert torch.equal(gqa_cache.get()[0], full_key) and torch.equal(gqa_cache.get()[1], full_value)
    check_attention("GQA Prefill + Decode", full_gqa_query, full_key, full_value, actual_gqa, expected_gqa)
    # 已完成 Prefill/Decode 的缓存不能再次 Prefill，拒绝后有效历史保持不变。
    try:
        prefill_attention(gqa_prefix, full_key[:, :, :3], full_value[:, :, :3], gqa_cache)
    except ValueError as error:
        assert gqa_cache.length == 4
        assert torch.equal(gqa_cache.get()[0], full_key) and torch.equal(gqa_cache.get()[1], full_value)
        print(f"[PASS] 非空缓存 Prefill 被拒绝，缓存未改变：{error}")
    else:
        raise AssertionError("[FAIL] Prefill 接口接受了非空缓存")
    # 3 个 Query 头不能分组到 2 个 KV 头：必须在追加前拒绝，历史内容保持不变。
    try:
        decode_attention(gqa_query[:, :3], full_key[:, :, 3:4], full_value[:, :, 3:4], gqa_cache)
    except ValueError as error:
        assert gqa_cache.length == 4
        assert torch.equal(gqa_cache.get()[0], full_key) and torch.equal(gqa_cache.get()[1], full_value)
        print(f"[PASS] 非法头数在写入前被拒绝，缓存未改变：{error}")
    else:
        raise AssertionError("[FAIL] Decode 接口接受了非法的头数关系")
    print("[PASS] CPU Prefill/Decode 接口自检通过；未验证 GPU、原生 GQA 内核或完整模型")


if __name__ == "__main__":
    main()
