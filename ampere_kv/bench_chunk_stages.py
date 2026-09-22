"""分块 Prefill 的逐块耗时探针：配对测读回、GQA 展开、掩码和整次 Attention 调用；不加载模型权重。

真实形状、不接常驻路径，也不参与验收：只回答"每个块额外付了哪几笔毫秒"，不回答"分块总共慢在哪"。
新旧两种读回写法在**同一次运行、同一份数据**上配对测量，避免跨轮次比较。
每段用 CUDA Event 包住并在测量后 synchronize，免得把异步派发的返回时间当成 GPU 完成时间。

口径（读数字前必看）：
- 列**不可相加**：`SDPA整调用*` 是完整的 `sdpa_attention()`，内部已经包含展开、掩码构造和
  `contiguous()`，与 `GQA展开`、`掩码` 两列重叠；它们只能逐列横向比较新旧或有无掩码。
- `SDPA整调用无偏移掩码` 那一列仍走同一条包装调用（`history=0` 时由 `is_causal=True` 承担因果，
  按左上角对齐），不是"去掉因果约束的裸 SDPA"；它与带掩码列的差值混着后端选择，不是纯掩码开销。
- 四块都调 `get()`，而生产首块 `history=0` 直接用本段 K/V、不读回，所以块1 的读回列在真实前向里不存在。
- "×36 层"是线性外推：不含投影、MLP、归一化与写入内核，也不含同步与宿主派发造成的 GPU 空闲；
  分段 Event 区间与模型级墙钟不是同一口径，不能互加减。端到端仍以 bench_scheduler 为准。
"""

import statistics

import torch

from ampere_kv.bench_scheduler import revision
from ampere_kv.kv_cache import sdpa_attention
from ampere_kv.paged_cache import PagedKVCache, PagedKVStorage

HEADS, Q_HEADS, DIM, SIZE, CHUNK, BLOCKS = 8, 32, 128, 16, 128, 4
GROUP = Q_HEADS // HEADS
REPEATS = 7
# 标签写死重叠关系：整调用两列包含展开与掩码，不与前面两列构成互斥分解。
STAGES = ("读回新", "读回旧", "GQA展开", "掩码", "SDPA整调用带掩码", "SDPA整调用无偏移掩码", "块表上传")


def timed(function):
    """返回 (中位数 ms, 极差)；先热身一次，再逐次记 CUDA Event 区间。"""
    function()
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
    values = []
    for _ in range(REPEATS):
        begin.record()
        result = function()
        end.record()
        torch.cuda.synchronize()
        values.append(begin.elapsed_time(end))
        del result
    return statistics.median(values), max(values) - min(values)


def old_read_back(cache, tensor):
    """改动前的写法：逐 Token 取块内位置切片再 `stack`，与当前实现配对比较。

    切片本身通常只是视图，主要开销在 K 与 V 各一次 L 路拼接；绝对毫秒含逐次同步，只用于横向比较斜率。
    """
    slots = [cache._table.locate(position) for position in range(cache.length)]
    return torch.stack([tensor[block, :, offset, :] for block, offset in slots], dim=1).unsqueeze(0)


def main():
    device = torch.cuda.get_device_properties(0)
    print(f"[环境] 设备={device.name}，显存={device.total_memory / 1024**3:.0f} GiB，"
          f"torch={torch.__version__}，代码版本={revision()}")
    print(f"[形状] KV头={HEADS}，Query头={Q_HEADS}，维度={DIM}，块={SIZE}，Chunk={CHUNK}，"
          f"块数={BLOCKS}，重复={REPEATS}")
    # 逐块追加、每块测"到此为止"的整段历史：形状与偏移对齐 bench_insert 的 512 Token 新请求。
    storage = PagedKVStorage(HEADS, DIM, CHUNK * BLOCKS // SIZE, SIZE, device="cuda")
    cache = PagedKVCache(storage)
    generator = torch.Generator(device="cuda").manual_seed(0)
    shape = (1, HEADS, CHUNK * BLOCKS, DIM)
    key = torch.randn(shape, generator=generator, device="cuda").to(torch.bfloat16)
    value = torch.randn(shape, generator=generator, device="cuda").to(torch.bfloat16)
    query = torch.randn((1, Q_HEADS, CHUNK, DIM), generator=generator, device="cuda").to(torch.bfloat16)
    totals = dict.fromkeys(STAGES, 0.0)
    try:
        for block in range(BLOCKS):
            history = block * CHUNK
            cache.append(key[:, :, history:history + CHUNK].contiguous(),
                         value[:, :, history:history + CHUNK].contiguous(), fused=True)
            # 读回含本块，长度即 cache.length；掩码偏移是 history，与 decoder_layer_forward 一致。
            # 生产首块 history=0 时不读回、直接用本段 K/V，所以块1 这一行给出的是探针值而非生产成本。
            new_ms, new_spread = timed(lambda: cache.get())
            old_ms, old_spread = timed(lambda: (old_read_back(cache, storage._key),
                                                old_read_back(cache, storage._value)))
            read_key, read_value = cache.get()
            expand_ms, _ = timed(lambda: (read_key.repeat_interleave(GROUP, dim=1),
                                          read_value.repeat_interleave(GROUP, dim=1)))
            mask_ms = 0.0
            if history:
                mask_ms, _ = timed(lambda: torch.arange(read_key.shape[2], device="cuda")[None, :]
                                   <= (history + torch.arange(CHUNK, device="cuda"))[:, None])
            masked_ms, _ = timed(lambda: sdpa_attention(query, read_key, read_value,
                                                        history=history, causal=True))
            # 同一包装调用去掉偏移掩码：仍由 is_causal 承担因果（按左上角对齐），差值里混着后端选择。
            plain_ms, _ = timed(lambda: sdpa_attention(query, read_key, read_value, history=0, causal=True))
            table_ms, _ = timed(lambda: torch.tensor(cache._table.block_ids, dtype=torch.long, device="cuda"))
            values = (new_ms, old_ms, expand_ms, mask_ms, masked_ms, plain_ms, table_ms)
            for name, ms in zip(STAGES, values):
                totals[name] += ms
            print(f"[块{block + 1}] 偏移={history} 长度={cache.length}："
                  f"读回新={new_ms:.2f}（极差{new_spread:.2f}）vs 旧={old_ms:.2f}（极差{old_spread:.2f}），"
                  f"GQA展开={expand_ms:.2f}，掩码={mask_ms:.2f}，SDPA整调用带掩码={masked_ms:.2f}，"
                  f"SDPA整调用无偏移掩码={plain_ms:.2f}，块表上传={table_ms:.2f}（单位 ms，整调用两列含展开与掩码）")
    finally:
        cache.release()
    print("[合计] " + "，".join(f"{name}={ms:.2f} ms" for name, ms in totals.items()))
    print("[边界] 整调用两列包含展开与掩码，与其他列重叠，任何一列都不能相加当总账；×36 是线性外推，"
          "不含投影/MLP/归一化与写入内核，也不含同步与派发留出的 GPU 空闲；块1 的读回列在真实前向里不存在；"
          "端到端仍以 bench_scheduler --only insert 为准。")


if __name__ == "__main__":
    main()
