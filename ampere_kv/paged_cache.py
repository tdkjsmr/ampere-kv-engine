"""单请求、单层 BF16 分页 K/V 与等头数 Decode 参考；尚未接入完整模型。"""

import torch

from ampere_kv.block_pool import BlockPool, BlockTable


class PagedKVCache:
    """固定大小的物理存储，按需分配块编号；当前每个实例独占一个块池。

    这是单线程参考实现，不支持多请求共享、量化或高性能批量写入。
    外部不能修改内部张量、块池或块表；输入 K/V 也不能与内部存储共享内存。
    """

    def __init__(self, num_kv_heads: int, head_dim: int, num_blocks: int,
                 block_size: int, device="cpu"):
        if any(type(size) is not int or size <= 0
               for size in (num_kv_heads, head_dim, num_blocks, block_size)):
            raise ValueError("KV 头数、每头维度、块数量和块大小必须是正整数")
        self._pool = BlockPool(num_blocks)
        self._table = BlockTable(self._pool, block_size)
        # K、V 各自布局为 [物理块数, KV 头数, 每块 Token 数, 每头维度]。
        # 一次分配全部物理空间；申请编号不会再次分配张量，空闲区域内容无效。
        shape = (num_blocks, num_kv_heads, block_size, head_dim)
        self._key = torch.empty(shape, dtype=torch.bfloat16, device=device)
        self._value = torch.empty(shape, dtype=torch.bfloat16, device=device)

    @property
    def length(self) -> int:
        """返回有效 Token 数；仅在 append 正常完成后才保证对应数据完整。"""
        return self._table.length

    @torch.no_grad()
    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """将 [1, KV 头数, 本次 Token 数, 每头维度] 的 BF16 K/V 写到尾部。"""
        # 主动拒绝形状、类型与设备不匹配，避免 copy_ 自动广播或转换掩盖错误。
        if key.ndim != 4 or key.shape != value.shape:
            raise ValueError("新 K/V 必须是形状相同的四维张量")
        if key.shape[0] != 1 or key.shape[1] != self._key.shape[1] or key.shape[3] != self._key.shape[3]:
            raise ValueError("新 K/V 的批大小、KV 头数或每头维度与缓存不一致")
        if key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16:
            raise ValueError("新 K/V 必须使用 BF16")
        if key.device != self._key.device or value.device != self._value.device:
            raise ValueError("新 K/V 必须与缓存位于同一设备")
        start = self.length
        # 块表拒绝空追加和容量不足；以上可预检错误均在修改存储前拒绝。
        # 先登记才能用 locate 查询新位置。若后续复制失败，不提供事务回滚；
        # 调用方应丢弃本次缓存状态，不能继续读取或盲目重试追加。
        self._table.append_tokens(key.shape[2])
        for index in range(key.shape[2]):
            physical_block, offset = self._table.locate(start + index)
            # 一次复制一个 Token 的全部 KV 头；这里只追求语义清晰，不追求速度。
            self._key[physical_block, :, offset, :].copy_(key[0, :, index, :])
            self._value[physical_block, :, offset, :].copy_(value[0, :, index, :])

    @torch.no_grad()
    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        """按逻辑顺序复制有效 K/V，返回 [1, KV 头数, 有效 Token 数, 每头维度]。

        返回的是独立副本，不是底层视图；后续高性能分页 Attention 不应这样读回历史。
        """
        if self.length == 0:
            shape = (1, self._key.shape[1], 0, self._key.shape[3])
            return self._key.new_empty(shape), self._value.new_empty(shape)
        keys, values = [], []
        for position in range(self.length):
            physical_block, offset = self._table.locate(position)
            keys.append(self._key[physical_block, :, offset, :])
            values.append(self._value[physical_block, :, offset, :])
        # 每个切片形状为 [KV 头数, 每头维度]，在中间插入 Token 维，再加批维。
        return torch.stack(keys, dim=1).unsqueeze(0), torch.stack(values, dim=1).unsqueeze(0)

    def release(self) -> None:
        """归还请求占用的块；保留物理张量，不清零旧数据，不释放底层显存。"""
        self._table.release()


@torch.no_grad()
def decode_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                     cache: PagedKVCache) -> torch.Tensor:
    """追加当前单 Token K/V，再用读回的连续历史计算 BF16 等头数 Attention。

    调用前缓存必须已有历史，且不能提前追加当前 K/V；本接口不支持 GQA。
    复制或计算失败后不回滚缓存；这不是直接访问分页存储的 CUDA 内核。
    """
    # Q 的检查必须在 append 前完成，防止非法 Query 已经改变缓存。
    if any(t.ndim != 4 for t in (query, key, value)):
        raise ValueError("Q/K/V 必须是四维张量")
    if query.shape != key.shape or key.shape != value.shape:
        raise ValueError("当前仅支持 Q/K/V 形状相同的等头数 Attention")
    if query.shape[0] != 1 or query.shape[2] != 1:
        raise ValueError("Decode 只支持单请求、单 Token")
    if query.dtype != torch.bfloat16 or query.device != key.device:
        raise ValueError("Query 必须使用 BF16 且与 K/V 位于同一设备")
    if cache.length == 0:
        raise ValueError("Decode 前必须已有历史 K/V")
    # append 继续检查 K/V 与物理存储的维度、精度、设备及剩余容量。
    cache.append(key, value)
    cached_key, cached_value = cache.get()
    # 当前 Query 对应最后一个有效位置，历史和当前 K/V 都可见，没有未来位置。
    # 不用 is_causal=True：单 Query 与长历史的非方形掩码可能屏蔽有效历史。
    return torch.nn.functional.scaled_dot_product_attention(
        query.contiguous(), cached_key.contiguous(), cached_value.contiguous(),
        dropout_p=0.0, is_causal=False, scale=query.shape[-1] ** -0.5,
    )


def main() -> None:
    """CPU 小张量自检：检查实际数据，不加载模型、不使用 GPU。"""
    cache = PagedKVCache(num_kv_heads=2, head_dim=4, num_blocks=3, block_size=4)
    assert cache.length == 0 and all(t.shape == (1, 2, 0, 4) for t in cache.get())
    addresses = (cache._key.data_ptr(), cache._value.data_ptr())
    # 自检专用：用哨兵填满物理存储，便于发现未使用位置被错误写入。
    cache._key.fill_(-100)
    cache._value.fill_(-200)
    # 通过块池正常操作构造 (2, 0, 1) 顺序，防止写入与读回都误用逻辑编号。
    held = [cache._pool.allocate() for _ in range(3)]
    for block_id in (held[1], held[0], held[2]):
        cache._pool.free(block_id)
    # 小整数可以由 BF16 精确表示；K/V 使用不同数值，便于发现混写。
    keys = torch.arange(72, dtype=torch.float32).reshape(1, 2, 9, 4).to(torch.bfloat16)
    values = keys + 128
    # 独立维护预期物理布局，不调用 locate，避免读写共用错误映射却相互抵消。
    expected_key, expected_value = cache._key.clone(), cache._value.clone()
    slots = ((2, 0), (2, 1), (2, 2), (2, 3), (0, 0),
             (0, 1), (0, 2), (0, 3), (1, 0))
    start = 0
    for count in (3, 2, 4):
        end = start + count
        cache.append(keys[:, :, start:end], values[:, :, start:end])
        for position in range(start, end):
            block_id, offset = slots[position]
            expected_key[block_id, :, offset] = keys[0, :, position]
            expected_value[block_id, :, offset] = values[0, :, position]
        assert cache.length == end
        assert torch.equal(cache._key, expected_key) and torch.equal(cache._value, expected_value)
        actual_key, actual_value = cache.get()
        assert actual_key.dtype == actual_value.dtype == torch.bfloat16
        assert torch.equal(actual_key, keys[:, :, :end]) and torch.equal(actual_value, values[:, :, :end])
        assert (cache._key.data_ptr(), cache._value.data_ptr()) == addresses
        print(f"[PASS] 分页追加 {count} 个 Token：长度={end}，物理布局与逻辑读回正确，地址不变")
        start = end
    assert cache._table.block_ids == (2, 0, 1)
    print("[PASS] 非连续块、跨块写入和末块无效尾部检查通过")

    before = (cache.length, cache._table.block_ids, cache._pool._free_blocks.copy(),
              cache._pool._allocated.copy())
    # 分别检查非法精度、空追加和超出容量；要求元数据及全部物理数据不变。
    for key, value, error_type in ((keys.float(), values, ValueError),
                                   (keys[:, :, :0], values[:, :, :0], ValueError),
                                   (keys[:, :, :4], values[:, :, :4], RuntimeError)):
        try:
            cache.append(key, value)
        except error_type:
            assert (cache.length, cache._table.block_ids, cache._pool._free_blocks,
                    cache._pool._allocated) == before
            assert torch.equal(cache._key, expected_key) and torch.equal(cache._value, expected_value)
        else:
            raise AssertionError("非法分页追加没有被拒绝")
    # 读回结果可修改，不得反向修改物理存储。
    actual_key.zero_()
    actual_value.zero_()
    assert torch.equal(cache._key, expected_key) and torch.equal(cache._value, expected_value)
    print("[PASS] 非法追加不改变状态，读回副本不影响底层存储")

    cache.release()
    assert cache.length == 0 and cache._pool.num_free_blocks == 3
    assert all(t.shape == (1, 2, 0, 4) for t in cache.get())
    # 使用不同数据重新开始，读回必须只有新请求的一个 Token，不得包含旧历史。
    cache.append(-keys[:, :, :1], -values[:, :, :1])
    actual_key, actual_value = cache.get()
    assert cache.length == 1
    assert torch.equal(actual_key, -keys[:, :, :1]) and torch.equal(actual_value, -values[:, :, :1])
    assert (cache._key.data_ptr(), cache._value.data_ptr()) == addresses
    cache.release()
    assert cache.length == 0 and cache._pool.num_free_blocks == 3
    assert not any(cache._pool._allocated)
    print("[PASS] 释放后可复用物理存储，读回不包含旧请求历史")
    print("[PASS] CPU 单层 BF16 分页 K/V 存储自检通过；未验证 Attention、多请求、模型或 GPU")

    # 独立 Decode 场景：先写 4 个历史 Token，再追加 1 个，恰好跨越块边界。
    cache = PagedKVCache(num_kv_heads=2, head_dim=4, num_blocks=3, block_size=4)
    cache._key.fill_(-100)
    cache._value.fill_(-200)
    held = [cache._pool.allocate() for _ in range(3)]
    for block_id in (held[1], held[0], held[2]):
        cache._pool.free(block_id)
    generator = torch.Generator().manual_seed(0)
    query = torch.randn(1, 2, 1, 4, generator=generator).to(torch.bfloat16)
    keys = torch.randn(1, 2, 5, 4, generator=generator).to(torch.bfloat16)
    values = torch.randn(1, 2, 5, 4, generator=generator).to(torch.bfloat16)
    cache.append(keys[:, :, :4], values[:, :, :4])
    addresses = (cache._key.data_ptr(), cache._value.data_ptr())
    before = (cache.length, cache._table.block_ids, cache._pool._free_blocks.copy(),
              cache._pool._allocated.copy())
    saved_key, saved_value = cache._key.clone(), cache._value.clone()
    # 非法头数和精度都必须在写入当前 Token 之前拒绝，保留全部原始状态。
    for invalid_query in (query[:, :1], query.float()):
        try:
            decode_attention(invalid_query, keys[:, :, 4:], values[:, :, 4:], cache)
        except ValueError:
            assert (cache.length, cache._table.block_ids, cache._pool._free_blocks,
                    cache._pool._allocated) == before
            assert torch.equal(cache._key, saved_key) and torch.equal(cache._value, saved_value)
        else:
            raise AssertionError("非法 Query 未在写入前被拒绝")
    print("[PASS] 非法 Decode Query 被拒绝，分页数据与元数据不变")
    actual = decode_attention(query, keys[:, :, 4:], values[:, :, 4:], cache)
    # 独立对照直接使用原始 K/V，不读缓存；精度、形状和 SDPA 参数保持相同。
    expected = torch.nn.functional.scaled_dot_product_attention(
        query.contiguous(), keys.contiguous(), values.contiguous(),
        dropout_p=0.0, is_causal=False, scale=query.shape[-1] ** -0.5,
    )
    assert actual.shape == (1, 2, 1, 4) and actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert cache.length == 5 and cache._table.block_ids == (2, 0)
    assert (cache._key.data_ptr(), cache._value.data_ptr()) == addresses
    actual_key, actual_value = cache.get()
    assert torch.equal(actual_key, keys) and torch.equal(actual_value, values)
    print("[PASS] 等头数 BF16 Decode：分页读回与原始连续 K/V 的 SDPA 输出完全一致")
    print("[PASS] Decode 跨块追加后长度=5，块表=(2, 0)，缓存内容正确且存储地址不变")
    cache.release()
    assert cache.length == 0 and cache._pool.num_free_blocks == 3
    print("[PASS] CPU 分页 Decode 参考自检通过；未验证 GQA、原生分页内核、模型或 GPU")


if __name__ == "__main__":
    main()
