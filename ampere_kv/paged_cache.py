"""单层 BF16 分页共享存储与请求缓存；共享生命周期仅做 CPU 参考验证。"""

import random

import torch

from ampere_kv.block_pool import BlockPool, BlockTable
# 复用同一套检查、缓存追加、GQA 头映射和 SDPA 计算，不另写分页包装。
# 分页 get() 会复制历史；这是参考路径，不是直接读取物理块的 CUDA 内核。
from ampere_kv.kv_cache import decode_attention, prefill_attention


class PagedKVStorage:
    """同一模型层的共享物理 K/V 与块池；所有请求使用相同布局和设备。"""

    def __init__(self, num_kv_heads: int, head_dim: int, num_blocks: int,
                 block_size: int, device="cpu"):
        if any(type(size) is not int or size <= 0
               for size in (num_kv_heads, head_dim, num_blocks, block_size)):
            raise ValueError("KV 头数、每头维度、块数量和块大小必须是正整数")
        self._pool = BlockPool(num_blocks)
        # K、V 各自布局为 [物理块数, KV 头数, 每块 Token 数, 每头维度]。
        # 一次分配全部物理空间；申请编号不会再次分配张量，空闲区域内容无效。
        shape = (num_blocks, num_kv_heads, block_size, head_dim)
        self._key = torch.empty(shape, dtype=torch.bfloat16, device=device)
        self._value = torch.empty(shape, dtype=torch.bfloat16, device=device)


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


def check_random_lifecycle() -> None:
    """固定种子的 1000 次 CPU 操作；最多四个活动请求，不测试并发线程。"""
    seed, operations, num_blocks, block_size = 0, 1000, 5, 4
    rng = random.Random(seed)
    generator = torch.Generator().manual_seed(seed)
    storage = PagedKVStorage(2, 4, num_blocks, block_size)
    storage._key.fill_(-100)
    storage._value.fill_(-200)
    # 每个请求同时保存独立的连续 K/V 作为预期数据，不从被测缓存构建参考。
    requests = {}
    next_id = 0
    counts = dict(create=0, append=0, release=0, rejected=0)
    addresses = (storage._key.data_ptr(), storage._value.data_ptr())

    def snapshot():
        # 复制元数据，避免引用同一列表导致前后比较失效；仅用于失败操作检查。
        return ([(rid, cache.length, cache._table.block_ids) for rid, (cache, _, _) in requests.items()],
                storage._pool._free_blocks.copy(), storage._pool._allocated.copy())

    for step in range(1, operations + 1):
        action, request_id, tokens = "create", None, 0
        try:
            if requests:
                # 追加权重较高，让小块池反复进入耗尽状态；不让活动请求数无限增长。
                choices = ["append", "append", "append", "release"]
                if len(requests) < 4:
                    choices.append("create")
                action = rng.choice(choices)
            if action == "create":
                request_id = next_id
                next_id += 1
                empty = torch.empty(1, 2, 0, 4, dtype=torch.bfloat16)
                requests[request_id] = (PagedKVCache(storage), empty, empty.clone())
                counts["create"] += 1
            else:
                request_id = rng.choice(list(requests))
                cache, expected_key, expected_value = requests[request_id]
                if action == "release":
                    cache.release()
                    assert cache.length == 0 and cache._table.block_ids == ()
                    del requests[request_id]
                    counts["release"] += 1
                else:
                    tokens = rng.randint(1, 7)
                    key = torch.randint(-64, 65, (1, 2, tokens, 4), generator=generator).to(torch.bfloat16)
                    value = torch.randint(-64, 65, (1, 2, tokens, 4), generator=generator).to(torch.bfloat16)
                    # 用独立参考长度预测容量，不相信被测块表或空闲计数给出的结果。
                    length = expected_key.shape[2]
                    used = sum((k.shape[2] + block_size - 1) // block_size for _, k, _ in requests.values())
                    needed = (length + tokens + block_size - 1) // block_size - (length + block_size - 1) // block_size
                    before = snapshot()
                    saved_key, saved_value = storage._key.clone(), storage._value.clone()
                    try:
                        cache.append(key, value)
                    except RuntimeError:
                        assert needed > num_blocks - used, "空间足够却拒绝追加"
                        assert snapshot() == before, "失败追加改变了元数据"
                        assert torch.equal(storage._key, saved_key) and torch.equal(storage._value, saved_value)
                        counts["rejected"] += 1
                    else:
                        assert needed <= num_blocks - used, "空间不足却接受追加"
                        requests[request_id] = (cache, torch.cat((expected_key, key), dim=2),
                                               torch.cat((expected_value, value), dim=2))
                        counts["append"] += 1

            # 每步检查所有活动请求，而非只检查被操作的请求，才能发现相互污染。
            occupied = []
            for cache, expected_key, expected_value in requests.values():
                assert cache.length == expected_key.shape[2]
                assert len(cache._table.block_ids) == (cache.length + block_size - 1) // block_size
                occupied.extend(cache._table.block_ids)
                actual_key, actual_value = cache.get()
                assert torch.equal(actual_key, expected_key) and torch.equal(actual_value, expected_value)
            free = storage._pool._free_blocks
            assert len(occupied) == len(set(occupied)), "请求占用块重复"
            assert len(free) == len(set(free)), "空闲块重复"
            assert set(occupied).isdisjoint(free), "已用块同时出现在空闲列表"
            assert set(occupied) | set(free) == set(range(num_blocks)), "块丢失或出现非法编号"
            assert storage._pool._allocated == [block in occupied for block in range(num_blocks)]
            assert (storage._key.data_ptr(), storage._value.data_ptr()) == addresses
        except Exception as error:
            raise AssertionError(f"随机自检失败：种子={seed}，操作序号={step}，操作={action}，请求={request_id}，追加数={tokens}") from error

    # 操作次数不等于请求数；必须实际覆盖创建、成功追加、释放和容量拒绝四类事件。
    assert sum(counts.values()) == operations and all(count > 0 for count in counts.values()), counts
    for cache, _, _ in requests.values():
        cache.release()
        assert cache.length == 0 and cache._table.block_ids == ()
    requests.clear()
    assert sorted(storage._pool._free_blocks) == list(range(num_blocks))
    assert not any(storage._pool._allocated)
    print(f"[PASS] 随机生命周期：种子={seed}，操作={operations}，创建请求={counts['create']}，成功追加={counts['append']}，释放={counts['release']}，容量拒绝={counts['rejected']}")
    print("[PASS] 每步数据与块归属检查通过，收尾全部块归还；不代表所有序列、并发线程或 GPU 验证通过")


def main() -> None:
    """CPU 小张量自检：检查实际数据，不加载模型、不使用 GPU。"""
    cache = PagedKVCache(PagedKVStorage(num_kv_heads=2, head_dim=4, num_blocks=3, block_size=4))
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

    # 等头数与 GQA 共用一个场景：Prefill 4 个 Token，Decode 跨块追加第 5 个。
    for label, mapping in (("等头数", [0, 1]), ("GQA", [0, 0, 1, 1])):
        cache = PagedKVCache(PagedKVStorage(num_kv_heads=2, head_dim=4, num_blocks=3, block_size=4))
        cache._key.fill_(-100)
        cache._value.fill_(-200)
        held = [cache._pool.allocate() for _ in range(3)]
        for block_id in (held[1], held[0], held[2]):
            cache._pool.free(block_id)
        generator = torch.Generator().manual_seed(0)
        query = torch.randn(1, len(mapping), 1, 4, generator=generator).to(torch.bfloat16)
        keys = torch.randn(1, 2, 5, 4, generator=generator).to(torch.bfloat16)
        values = torch.randn(1, 2, 5, 4, generator=generator).to(torch.bfloat16)
        prefix_query = torch.randn(1, len(mapping), 4, 4, generator=generator).to(torch.bfloat16)
        # 两阶段对照都从原始 K/V 显式选头，不读缓存、不复用 repeat_interleave。
        head_indices = torch.tensor(mapping, dtype=torch.long)
        reference_key = keys.index_select(1, head_indices)
        reference_value = values.index_select(1, head_indices)
        # 合并后仍须拒绝空历史 Decode，且不能偷偷写入当前 K/V。
        try:
            decode_attention(query, keys[:, :, :1], values[:, :, :1], cache)
        except ValueError:
            assert cache.length == 0 and cache._pool.num_free_blocks == 3
            assert cache._table.block_ids == ()
            assert (cache._key == -100).all() and (cache._value == -200).all()
        else:
            raise AssertionError("空历史 Decode 未被拒绝")
        addresses = (cache._key.data_ptr(), cache._value.data_ptr())
        # 共用 Prefill 自己负责写入 K/V，不能在调用前再 append 一次。
        prefix_output = prefill_attention(prefix_query, keys[:, :, :4], values[:, :, :4], cache)
        expected_prefix = torch.nn.functional.scaled_dot_product_attention(
            prefix_query.contiguous(), reference_key[:, :, :4].contiguous(),
            reference_value[:, :, :4].contiguous(), dropout_p=0.0, is_causal=True,
            scale=prefix_query.shape[-1] ** -0.5,
        )
        assert prefix_output.shape == (1, len(mapping), 4, 4)
        assert prefix_output.dtype == torch.bfloat16 and torch.isfinite(prefix_output).all()
        torch.testing.assert_close(prefix_output, expected_prefix, rtol=0, atol=0)
        assert cache.length == 4 and cache._table.block_ids == (2,)
        actual_key, actual_value = cache.get()
        assert torch.equal(actual_key, keys[:, :, :4]) and torch.equal(actual_value, values[:, :, :4])
        assert (cache._key.data_ptr(), cache._value.data_ptr()) == addresses
        print(f"[PASS] {label} BF16 Prefill：因果输出完全一致，长度=4，块表=(2,)")
        before = (cache.length, cache._table.block_ids, cache._pool._free_blocks.copy(),
                  cache._pool._allocated.copy())
        saved_key, saved_value = cache._key.clone(), cache._value.clone()
        # 当前 Prefill 只允许空缓存；拒绝后必须保留刚写入的全部数据和元数据。
        try:
            prefill_attention(prefix_query, keys[:, :, :4], values[:, :, :4], cache)
        except ValueError:
            assert (cache.length, cache._table.block_ids, cache._pool._free_blocks,
                    cache._pool._allocated) == before
            assert torch.equal(cache._key, saved_key) and torch.equal(cache._value, saved_value)
        else:
            raise AssertionError("非空分页缓存接受了重复 Prefill")
        print(f"[PASS] {label}：非空缓存 Prefill 被拒绝，分页数据与元数据不变")
        # 非法头数和精度都必须在写入当前 Token 之前拒绝，保留全部原始状态。
        for invalid_query in (query[:, :1], query[:, :1].expand(1, 3, 1, 4), query.float()):
            try:
                decode_attention(invalid_query, keys[:, :, 4:], values[:, :, 4:], cache)
            except ValueError:
                assert (cache.length, cache._table.block_ids, cache._pool._free_blocks,
                        cache._pool._allocated) == before
                assert torch.equal(cache._key, saved_key) and torch.equal(cache._value, saved_value)
            else:
                raise AssertionError("非法 Query 未在写入前被拒绝")
        print(f"[PASS] {label}：非法 Decode Query 被拒绝，分页数据与元数据不变")
        actual = decode_attention(query, keys[:, :, 4:], values[:, :, 4:], cache)
        # 最后一个 Query 可见全部 5 个有效 K/V，与 Prefill 的因果设置不同。
        expected = torch.nn.functional.scaled_dot_product_attention(
            query.contiguous(), reference_key.contiguous(), reference_value.contiguous(),
            dropout_p=0.0, is_causal=False, scale=query.shape[-1] ** -0.5,
        )
        assert actual.shape == (1, len(mapping), 1, 4) and actual.dtype == torch.bfloat16
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert cache.length == 5 and cache._table.block_ids == (2, 0)
        assert (cache._key.data_ptr(), cache._value.data_ptr()) == addresses
        # Query 头增加只影响参考计算的临时数据，物理缓存仍然只有两个 KV 头。
        assert cache._key.shape == cache._value.shape == (3, 2, 4, 4)
        actual_key, actual_value = cache.get()
        assert torch.equal(actual_key, keys) and torch.equal(actual_value, values)
        print(f"[PASS] {label} BF16 Decode：头映射={mapping}，输出形状={tuple(actual.shape)}，与原始 K/V 对照完全一致")
        print(f"[PASS] {label}：跨块后长度=5，块表=(2, 0)，只存两个 KV 头且地址不变")
        cache.release()
        assert cache.length == 0 and cache._pool.num_free_blocks == 3
        assert not any(cache._pool._allocated)
    print("[PASS] CPU 分页等头数/GQA Prefill + Decode 自检通过；未验证原生分页内核、多请求、模型或 GPU")

    # 两个请求交错占用同一层存储，不运行多请求 Attention 或调度器。
    storage = PagedKVStorage(num_kv_heads=2, head_dim=4, num_blocks=3, block_size=4)
    storage._key.fill_(-100)
    storage._value.fill_(-200)
    a, b = PagedKVCache(storage), PagedKVCache(storage)
    assert a._key is b._key is storage._key and a._value is b._value is storage._value
    assert a._pool is b._pool is storage._pool and a._table is not b._table
    keys = torch.arange(40, dtype=torch.float32).reshape(1, 2, 5, 4).to(torch.bfloat16)
    a.append(keys[:, :, :3], keys[:, :, :3] + 64)
    b.append(-keys[:, :, :3], -keys[:, :, :3] - 64)
    a.append(keys[:, :, 3:], keys[:, :, 3:] + 64)
    assert a._table.block_ids == (0, 2) and b._table.block_ids == (1,)
    assert set(a._table.block_ids).isdisjoint(b._table.block_ids)
    assert a.length == 5 and b.length == 3 and storage._pool.num_free_blocks == 0
    for actual, expected in zip(a.get(), (keys, keys + 64)):
        assert torch.equal(actual, expected)
    for actual, expected in zip(b.get(), (-keys[:, :, :3], -keys[:, :, :3] - 64)):
        assert torch.equal(actual, expected)
    print("[PASS] 两请求共享物理张量，交错追加后块不重叠、数据互不混写")

    # 总容量为 12，但其他请求占用了块；B 从 3 增至 5 需要新块，必须失败。
    saved_key, saved_value = storage._key.clone(), storage._value.clone()
    before = (a.length, a._table.block_ids, b.length, b._table.block_ids,
              storage._pool._free_blocks.copy(), storage._pool._allocated.copy())
    try:
        b.append(-keys[:, :, 3:], -keys[:, :, 3:] - 64)
    except RuntimeError:
        assert (a.length, a._table.block_ids, b.length, b._table.block_ids,
                storage._pool._free_blocks, storage._pool._allocated) == before
        assert torch.equal(storage._key, saved_key) and torch.equal(storage._value, saved_value)
    else:
        raise AssertionError("共享块池耗尽时仍允许跨块追加")
    print("[PASS] 共享池耗尽时拒绝追加，两个请求与物理数据均不变")

    released = set(a._table.block_ids)
    a.release()
    assert a.length == 0 and a._table.block_ids == () and storage._pool.num_free_blocks == 2
    assert all(t.shape == (1, 2, 0, 4) for t in a.get())
    assert b.length == 3 and b._table.block_ids == (1,)
    c = PagedKVCache(storage)
    c.append(keys[:, :, :1] + 128, keys[:, :, :1] + 192)
    assert set(c._table.block_ids).issubset(released) and c.length == 1
    for actual, expected in zip(c.get(), (keys[:, :, :1] + 128, keys[:, :, :1] + 192)):
        assert torch.equal(actual, expected)
    a.release()  # A 的旧块已被 C 复用，重复清理 A 不能误释放 C 的块。
    assert all(storage._pool._allocated[block] for block in c._table.block_ids)
    b.append(-keys[:, :, 3:], -keys[:, :, 3:] - 64)
    for actual, expected in zip(b.get(), (-keys, -keys - 64)):
        assert torch.equal(actual, expected)
    for actual, expected in zip(c.get(), (keys[:, :, :1] + 128, keys[:, :, :1] + 192)):
        assert torch.equal(actual, expected)
    assert set(b._table.block_ids).isdisjoint(c._table.block_ids)
    b.release()
    c.release()
    assert storage._pool.num_free_blocks == 3 and not any(storage._pool._allocated)
    assert b.length == c.length == 0 and b._table.block_ids == c._table.block_ids == ()
    print("[PASS] 释放 A 后新请求复用旧块，B 可继续追加且数据隔离，最终全部块归还")
    print("[PASS] CPU 共享分页存储生命周期自检通过；未验证并发线程、多请求模型或 GPU")
    check_random_lifecycle()


if __name__ == "__main__":
    main()
