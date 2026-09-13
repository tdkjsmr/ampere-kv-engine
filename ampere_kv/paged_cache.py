"""单层 BF16/INT8 分页存储；CUDA 读取与当前数值对照位于 paged_decode。"""

import torch

from ampere_kv.block_pool import BlockPool, BlockTable
from ampere_kv.quantization import quantize_kv, dequantize_kv
# 分页 get() 会复制历史；这是参考路径，不是直接读取物理块的 CUDA 内核。


class PagedKVStorage:
    """同一模型层的共享物理 K/V 与块池；所有请求使用相同布局和设备。"""

    def __init__(self, num_kv_heads: int, head_dim: int, num_blocks: int,
                 block_size: int, device="cpu", *, kv_dtype=torch.bfloat16):
        if any(type(size) is not int or size <= 0
               for size in (num_kv_heads, head_dim, num_blocks, block_size)):
            raise ValueError("KV 头数、每头维度、块数量和块大小必须是正整数")
        if kv_dtype not in (torch.bfloat16, torch.int8):
            raise ValueError("分页存储只支持 BF16 或 INT8")
        if kv_dtype == torch.int8 and head_dim != 128:
            raise ValueError("当前 INT8 存储固定每头128维，K分成四组32维")
        self._pool = BlockPool(num_blocks)
        # K、V 各自布局为 [物理块数, KV 头数, 每块 Token 数, 每头维度]。
        # 一次分配全部物理空间；申请编号不会再次分配张量，空闲区域内容无效。
        shape = (num_blocks, num_kv_heads, block_size, head_dim)
        self._key = torch.empty(shape, dtype=kv_dtype, device=device)
        self._value = torch.empty(shape, dtype=kv_dtype, device=device)
        # INT8 每个物理位置、每个 KV 头分别保存 K/V scale，不常驻 BF16 历史副本。
        self._key_scale = self._value_scale = None
        if kv_dtype == torch.int8:
            # K 每连续32维一个 scale；V 仍是整个128维一个 scale。
            self._key_scale = torch.empty(shape[:-1] + (4,), dtype=torch.float16, device=device)
            self._value_scale = torch.empty(shape[:-1] + (1,), dtype=torch.float16, device=device)


class PagedKVCache:
    """一个请求的块表与共享存储引用；已分配的块仍由该请求独占。

    单线程使用，不支持共享同一已分配块、引用计数或前缀复用。
    请求结束必须显式 release；丢弃请求对象不会自动向共享池归还编号。
    外部不能修改内部状态，输入 K/V 不能与物理存储共享内存。
    """

    def __init__(self, storage: PagedKVStorage):
        # 这里只复制 Python 引用，不分配或复制 K/V 张量，也不重新创建块池。
        self._pool = storage._pool
        self._key, self._value = storage._key, storage._value
        self._key_scale, self._value_scale = storage._key_scale, storage._value_scale
        self._table = BlockTable(self._pool, self._key.shape[2])

    @property
    def capacity(self) -> int:
        """返回整个共享池的 Token 容量，不保证本请求可全部占用；追加另查空闲块。"""
        return self._key.shape[0] * self._key.shape[2]

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
        if self._key_scale is not None:
            # K/V 全部量化成功后才申请块；例如 V 含 NaN 时，不能留下只写入 K 的状态。
            # 这里只产生本次追加数据的临时结果，不重新量化旧历史。
            shape = key.shape
            # 临时合并 Token 与组维，只沿32维量化；不混合不同 Token 的分量。
            key, key_scale = quantize_kv(key.reshape(1, shape[1], shape[2] * 4, 32))
            key = key.reshape(shape)
            key_scale = key_scale.reshape(1, shape[1], shape[2], 4)
            value, value_scale = quantize_kv(value)
        # 块表拒绝空追加和容量不足；以上可预检错误均在修改存储前拒绝。
        # 先登记才能用 locate 查询新位置。若后续复制失败，不提供事务回滚；
        # 调用方应丢弃本次缓存状态，不能继续读取或盲目重试追加。
        self._table.append_tokens(key.shape[2])
        for index in range(key.shape[2]):
            physical_block, offset = self._table.locate(start + index)
            # 一次复制一个 Token 的全部 KV 头；这里只追求语义清晰，不追求速度。
            self._key[physical_block, :, offset, :].copy_(key[0, :, index, :])
            self._value[physical_block, :, offset, :].copy_(value[0, :, index, :])
            if self._key_scale is not None:
                self._key_scale[physical_block, :, offset, :].copy_(key_scale[0, :, index, :])
                self._value_scale[physical_block, :, offset, :].copy_(value_scale[0, :, index, :])

    @torch.no_grad()
    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        """按逻辑顺序复制有效 K/V，返回 [1, KV 头数, 有效 Token 数, 每头维度]。

        BF16 返回 BF16；INT8 读取有效整数和 scale 后还原为 FP32，供参考 Attention 使用。
        返回独立副本，不是底层视图；高性能分页 Attention 不应这样读回历史。
        """
        if self.length == 0:
            shape = (1, self._key.shape[1], 0, self._key.shape[3])
            dtype = torch.float32 if self._key_scale is not None else torch.bfloat16
            return self._key.new_empty(shape, dtype=dtype), self._value.new_empty(shape, dtype=dtype)
        # 数据与 scale 共用一份有效位置列表，末块未写入部分不会被读取。
        slots = [self._table.locate(position) for position in range(self.length)]

        def gather(tensor):
            return torch.stack([tensor[block, :, offset, :] for block, offset in slots], dim=1).unsqueeze(0)

        key, value = gather(self._key), gather(self._value)
        if self._key_scale is not None:
            # 按相同顺序拆组反量化，再还原每个 Token 的128维；不展开 GQA 头。
            grouped_key = key.reshape(1, key.shape[1], self.length * 4, 32)
            scales = gather(self._key_scale).reshape(1, key.shape[1], self.length * 4, 1)
            return (dequantize_kv(grouped_key, scales).reshape(key.shape),
                    dequantize_kv(value, gather(self._value_scale)))
        return key, value

    def release(self) -> None:
        """归还请求占用的块；保留物理张量，不清零旧数据，不释放底层显存。"""
        self._table.release()


def main() -> None:
    """本轮仅验证分组K的物理布局与读回；CPU执行，不调用CUDA或模型。"""
    storage = PagedKVStorage(2, 128, 2, 16, kv_dtype=torch.int8)
    cache = PagedKVCache(storage)
    tensors = (cache._key, cache._value, cache._key_scale, cache._value_scale)
    addresses = tuple(t.data_ptr() for t in tensors)
    # 先归还0再归还1，令逻辑块顺序为(1, 0)，检查跨块寻址而非连续巧合。
    held = [storage._pool.allocate() for _ in range(2)]
    for block in held:
        storage._pool.free(block)
    generator = torch.Generator().manual_seed(3)
    key = torch.randn(1, 2, 17, 128, generator=generator)
    key *= torch.tensor([1., 10., 0.1, 100.]).repeat_interleave(32)
    key = key.to(torch.bfloat16)
    value = torch.randn(1, 2, 17, 128, generator=generator).to(torch.bfloat16)
    # 独立按切片逐组量化，避免参考也用同一 reshape 顺序而掩盖布局错误。
    groups = [quantize_kv(key[..., start:start + 32]) for start in range(0, 128, 32)]
    kd = torch.cat([data for data, scale in groups], dim=-1)
    ks = torch.cat([scale for data, scale in groups], dim=-1)
    vd, vs = quantize_kv(value)
    expected_key = torch.cat([data.float() * scale.float() for data, scale in groups], dim=-1)
    try:
        for start, end in ((0, 15), (15, 17)):
            cache.append(key[:, :, start:end], value[:, :, start:end])
            for tensor, reference in zip(tensors, (kd, vd, ks, vs)):
                for token in range(end):
                    block, offset = cache._table.locate(token)
                    assert torch.equal(tensor[block, :, offset], reference[0, :, token])
            actual_key, actual_value = cache.get()
            torch.testing.assert_close(actual_key, expected_key[:, :, :end], rtol=0, atol=0)
            torch.testing.assert_close(actual_value, (vd.float() * vs.float())[:, :, :end], rtol=0, atol=0)
            assert cache.length == end and tuple(t.data_ptr() for t in tensors) == addresses
            print(f"[PASS] 分组K存储：长度={end}，整数与scale布局、FP32读回及地址一致")
        assert cache._table.block_ids == (1, 0)
    finally:
        cache.release()
    assert storage._pool.num_free_blocks == 2
    print("[PASS] CPU分组K存储与跨块追加通过；K scale末维4，V末维1；未验证新布局CUDA、模型或性能")


if __name__ == "__main__":
    main()
