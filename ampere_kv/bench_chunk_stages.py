"""阶段归因：分块 Prefill 每块的读回/展开/掩码/SDPA 耗时配对；不加载模型权重。

真实形状、不接常驻路径，也不参与验收：只回答"每个块多付的那笔开销落在哪一段"。
新旧两种读回写法在**同一次运行、同一份数据**上配对测量，避免跨轮次比较。
每段用 CUDA Event 包住并在测量后 synchronize，免得把异步派发的返回时间当成 GPU 完成时间。
"×36 层"只是线性外推，不是端到端实测；端到端仍以 bench_scheduler 为准。
"""

import statistics

import torch

from ampere_kv.bench_scheduler import revision
from ampere_kv.kv_cache import sdpa_attention
from ampere_kv.paged_cache import PagedKVCache, PagedKVStorage

HEADS, Q_HEADS, DIM, SIZE, CHUNK, BLOCKS = 8, 32, 128, 16, 128, 4
GROUP = Q_HEADS // HEADS
REPEATS = 7
STAGES = ("读回新", "读回旧", "GQA展开", "掩码", "SDPA带掩码", "SDPA无掩码", "块表上传")


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
    """改动前的写法：逐 Token 切片再 stack，用来在同一份数据上与新实现配对比较。"""
    slots = [cache._table.locate(position) for position in range(cache.length)]
    return torch.stack([tensor[block, :, offset, :] for block, offset in slots], dim=1).unsqueeze(0)


def main():
    device = torch.cuda.get_device_properties(0)
    print(f"[环境] 设备={device.name}，显存={device.total_memory / 1024**3:.0f} GiB，"
          f"torch={torch.__version__}，代码版本={revision()}")
    print(f"[形状] KV头={HEADS}，Query头={Q_HEADS}，维度={DIM}，块={SIZE}，Chunk={CHUNK}，"
          f"块数={BLOCKS}，重复={REPEATS}")
    # 先填满整个池再逐块读回，令每块的偏移与 bench_insert 的 512 Token 新请求一致。
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
            # 同形状去掉偏移掩码：语义不成立（torch 的 is_causal 按左上角对齐），只观察后端选择差异。
            plain_ms, _ = timed(lambda: sdpa_attention(query, read_key, read_value, history=0, causal=True))
            table_ms, _ = timed(lambda: torch.tensor(cache._table.block_ids, dtype=torch.long, device="cuda"))
            values = (new_ms, old_ms, expand_ms, mask_ms, masked_ms, plain_ms, table_ms)
            for name, ms in zip(STAGES, values):
                totals[name] += ms
            print(f"[块{block + 1}] 偏移={history} 长度={cache.length}："
                  f"读回新={new_ms:.2f}（极差{new_spread:.2f}）vs 旧={old_ms:.2f}（极差{old_spread:.2f}），"
                  f"GQA展开={expand_ms:.2f}，掩码={mask_ms:.2f}，SDPA带掩码={masked_ms:.2f}，"
                  f"SDPA无掩码={plain_ms:.2f}，块表上传={table_ms:.2f}（单位 ms）")
    finally:
        cache.release()
    print("[合计] " + "，".join(f"{name}={ms:.2f} ms" for name, ms in totals.items()))
    print("[边界] 每层毫秒数 ×36 是线性外推，不含投影/MLP/归一化与写入内核；SDPA无掩码只是后端对照列，"
          "不是可用实现；端到端仍以 bench_scheduler --only insert 为准。")


if __name__ == "__main__":
    main()
