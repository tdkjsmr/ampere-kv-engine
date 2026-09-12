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
        self._pool = BlockPool(num_blocks)
        # K、V 各自布局为 [物理块数, KV 头数, 每块 Token 数, 每头维度]。
        # 一次分配全部物理空间；申请编号不会再次分配张量，空闲区域内容无效。
        shape = (num_blocks, num_kv_heads, block_size, head_dim)
        self._key = torch.empty(shape, dtype=kv_dtype, device=device)
        self._value = torch.empty(shape, dtype=kv_dtype, device=device)
        # INT8 每个物理位置、每个 KV 头分别保存 K/V scale，不常驻 BF16 历史副本。
        self._key_scale = self._value_scale = None
        if kv_dtype == torch.int8:
            self._key_scale = torch.empty(shape[:-1] + (1,), dtype=torch.float16, device=device)
            self._value_scale = torch.empty_like(self._key_scale)


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
            key, key_scale = quantize_kv(key)
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
            return (dequantize_kv(key, gather(self._key_scale)),
                    dequantize_kv(value, gather(self._value_scale)))
        return key, value

    def release(self) -> None:
        """归还请求占用的块；保留物理张量，不清零旧数据，不释放底层显存。"""
        self._table.release()
