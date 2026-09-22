"""固定 Chunk 验收：偏移掩码、真实模型对照和交错调度；不在生产热路径放检查。"""

import torch

from ampere_kv.kv_cache import sdpa_attention
from ampere_kv.paged_cache import PagedKVCache, PagedKVStorage
from ampere_kv.runner import encode_prompt, load_model_and_tokenizer, model_forward
from ampere_kv.scheduler import FINISHED, Scheduler


def check_mask():
    """同一份 FP32 Q/K/V 对独立 FP64 全序列公式，覆盖历史偏移和单 Token 尾块。"""
    generator = torch.Generator().manual_seed(7)
    query = torch.randn(1, 4, 65, 8, generator=generator)
    key = torch.randn(1, 2, 65, 8, generator=generator)
    value = torch.randn(1, 2, 65, 8, generator=generator)
    scores = query.double() @ key.double().repeat_interleave(2, dim=1).transpose(-2, -1) / 8 ** 0.5
    future = torch.ones(65, 65, dtype=torch.bool).triu(1)
    expected = scores.masked_fill(future, -torch.inf).softmax(-1) @ value.double().repeat_interleave(2, dim=1)
    for start, end in ((0, 17), (17, 64), (64, 65)):
        actual = sdpa_attention(query[:, :, start:end], key[:, :, :end], value[:, :, :end],
                                history=start, causal=True)
        torch.testing.assert_close(actual.double(), expected[:, :, start:end], rtol=1e-5, atol=1e-5)
    print("[PASS] 偏移因果掩码对独立公式通过：17/47/1 Token，含 GQA 与单 Token 尾块")


@torch.inference_mode()
def check_read_back():
    """BF16 批量读回与 locate 逐位置参考逐位一致，并确认副本、非连续块序与部分尾块。

    这是对"换一种索引写法"的等价性检查，不是新的测试框架：只覆盖分块 Prefill 真正依赖的
    那条读回路径，INT8 分支本轮没改所以不在此处重复验证。
    """
    heads, size, dim, blocks = 2, 16, 128, 6
    storage = PagedKVStorage(heads, dim, blocks, size, device="cuda")
    storage._key.fill_(float("nan"))
    storage._value.fill_(float("nan"))
    cache = PagedKVCache(storage)
    # 按分配顺序归还，让后取到的物理编号倒序：逻辑顺序 0,1,2 对应物理 4,3,2。
    held = [storage._pool.allocate() for _ in range(5)]
    for block in held:
        storage._pool.free(block)
    generator = torch.Generator(device="cuda").manual_seed(5)
    key = torch.randn(1, heads, 33, dim, generator=generator, device="cuda").to(torch.bfloat16)
    value = torch.randn(1, heads, 33, dim, generator=generator, device="cuda").to(torch.bfloat16)
    try:
        for start, end in ((0, 16), (16, 32), (32, 33)):  # 33 Token 只用掉最后一块的第 1 格
            cache.append(key[:, :, start:end], value[:, :, start:end])
        assert cache._table.block_ids == (4, 3, 2), "用例前提被破坏：块表不是非连续顺序"
        read_key, read_value = cache.get()
        slots = [cache._table.locate(position) for position in range(33)]
        for got, source, original in ((read_key, storage._key, key), (read_value, storage._value, value)):
            assert got.shape == (1, heads, 33, dim)
            want = torch.stack([source[block, :, offset] for block, offset in slots], dim=1).unsqueeze(0)
            torch.testing.assert_close(got, want, rtol=0, atol=0)      # 与 locate 定义逐位一致
            torch.testing.assert_close(got, original, rtol=0, atol=0)  # 还原成逻辑顺序即输入顺序
            before = source.clone()
            got.fill_(0.0)                                            # 读回必须是独立副本
            torch.testing.assert_close(source, before, rtol=0, atol=0, equal_nan=True)
        tail_block, tail_offset = cache._table.locate(32)
        assert torch.isnan(storage._key[tail_block, :, tail_offset + 1:]).all(), "尾块未写部分被写过"
    finally:
        cache.release()
    print("[PASS] BF16 批量读回：非连续块表、部分尾块、独立副本与 locate 逐位置参考一致")


def check_model(model, prompt, chunk=128):
    """整段/分块各自完成 Prefill 和短 Decode；logits 报误差，不预设跨形状逐位一致。"""
    length, new_tokens = prompt.shape[1], 8
    blocks = (length + new_tokens + 15) // 16
    cache_sets, logits = [], []
    try:
        for size in (length, chunk):
            owner = Scheduler(model, blocks)
            caches = [PagedKVCache(storage) for storage in owner.storages]
            cache_sets.append(caches)
            for start in range(0, length, size):
                end = min(start + size, length)
                positions = torch.arange(start, end, device=prompt.device).unsqueeze(0)
                out = model_forward(model, prompt[:, start:end], positions, caches,
                                    is_prefill=True, output_logits=end == length)
                assert all(cache.length == end for cache in caches)
                if end < length:
                    assert out is None, "中间块不应产生 logits"
            logits.append(out)
        sequences = [[], []]
        for step in range(new_tokens):
            reference, actual = (tensor.float() for tensor in logits)
            assert torch.isfinite(reference).all() and torch.isfinite(actual).all()
            difference = actual - reference
            relative = difference.norm() / reference.norm().clamp_min(1e-12)
            ids = [tensor.argmax(-1).item() for tensor in logits]
            print(f"[观测] 输出{step + 1}：logits最大误差={difference.abs().max().item():.6g}，"
                  f"相对L2={relative.item():.6g}，整段/分块Token={ids}")
            assert ids[0] == ids[1], "整段/分块选词分歧，先定位；不自动放宽验收"
            for sequence, token in zip(sequences, ids):
                sequence.append(token)
            if step + 1 < new_tokens:
                position = torch.tensor([[length + step]], device=prompt.device)
                logits = [model_forward(model, prompt.new_tensor([[token]]), position, caches,
                                        is_prefill=False, cuda_decode=True)
                          for token, caches in zip(ids, cache_sets)]
        print("[PASS] 整段/分块短序列一致；logits仅观测，固定8个Token忽略EOS，不是完整精度验收")
        return sequences[1]
    finally:
        for caches in cache_sets:
            for cache in caches:
                cache.release()
            assert caches[0]._pool.num_free_blocks == blocks


def check_interleaved(model, prompt, expected, chunk=128):
    """短请求先 Decode，长请求分三块接入；独立分块序列是长请求的对照。"""
    scheduler = Scheduler(model, 32, max_batch=1, prefill_chunk_size=chunk)
    short = scheduler.submit("short", prompt[:, :17], 8, ignore_eos=True)
    scheduler.step()
    long = scheduler.submit("long", prompt, 8, ignore_eos=True)
    for _ in range((prompt.shape[1] + chunk - 1) // chunk):
        previous = len(short.output_ids)
        scheduler.step()
        assert len(short.output_ids) == previous + 1, "长请求 Prefill 阻塞了短请求的轮次推进"
        if scheduler.prefilling is not None:
            assert not long.output_ids and long not in scheduler.running
    while scheduler.running or scheduler.prefilling is not None or scheduler.waiting:
        scheduler.step()
    assert long.status == short.status == FINISHED and long.output_ids == expected
    # 短请求也对独立运行，防止只检查推进数量而漏掉跨请求混写。
    from ampere_kv.runner import generate_tokens
    assert short.output_ids == generate_tokens(model, prompt[:, :17], max_new_tokens=8,
                                               cache_kind="paged", cuda_decode=True, ignore_eos=True)
    assert scheduler.reserved_blocks == 0
    assert all(storage._pool.num_free_blocks == 32 for storage in scheduler.storages)
    print("[PASS] 分块期间 Decode 持续推进、未提前选词、交错序列与独立一致，全部块归还")


def main():
    check_mask()
    check_read_back()          # 两条不加载模型的检查排在前面，失败时不必先等 16 GB 权重
    model, tokenizer = load_model_and_tokenizer()
    ids = encode_prompt(tokenizer, "用一句话解释 KV 缓存。")
    prompt = ids.repeat(1, (257 + ids.shape[1] - 1) // ids.shape[1])[:, :257]
    print("[配置] BF16/V1；输入257 Token，Chunk=128，尾块1 Token")
    expected = check_model(model, prompt)
    check_interleaved(model, prompt, expected)
    print("[完成] 固定 Chunk 语义与调度验收；未验证取消、超时、SLO、Graph 或性能")


if __name__ == "__main__":
    main()
