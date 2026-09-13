"""CUDA Paged Decode V0 对照与可选调用基线；不加载模型。"""

import argparse
import statistics
import time

import torch

from ampere_kv import _C
from ampere_kv.paged_cache import PagedKVCache, PagedKVStorage
from ampere_kv.quantization import quantize_kv


@torch.inference_mode()
def benchmark() -> None:
    """固定数据上的算子调用墙钟基线；不包含分页整理，不等于纯内核耗时。"""
    generator = torch.Generator().manual_seed(0)
    warmup, repeats, iterations = 5, 3, 20
    print(f"调用基线：Q头=32，KV头=8，每头128维，块大小16；预热={warmup} 次，测量={repeats} 组，每组={iterations} 次")
    print("计时前准备 Q/K/V、GQA 展开和 GPU 块表；计时包含 Python/扩展调用、输出分配、CUDA V0 块号检查同步及组末等待。")
    print("同一份数据反复读取，可能受硬件缓存影响；SDPA 后端自动选择，本结果不是纯内核时间、请求 TPOT 或原生 GQA 性能。")
    print("CUDA 段长=64；长度超过 64 时启用分段，临时结果分配与合并调用均计时；修改段长后须重新编译扩展。")
    for length in (64, 256, 1024):
        query = torch.randn(1, 32, 1, 128, generator=generator).to(device="cuda", dtype=torch.bfloat16)
        key = torch.randn(1, 8, length, 128, generator=generator).to(device="cuda", dtype=torch.bfloat16)
        value = torch.randn(1, 8, length, 128, generator=generator).to(device="cuda", dtype=torch.bfloat16)
        storage = PagedKVStorage(8, 128, length // 16, 16, device="cuda")
        cache = PagedKVCache(storage)
        try:
            # 倒序归还块号，使物理顺序不同于逻辑顺序；准备与追加均不计时。
            held = [storage._pool.allocate() for _ in range(length // 16)]
            for block in held:
                storage._pool.free(block)
            cache.append(key, value)
            table = torch.tensor(cache._table.block_ids, dtype=torch.long, device="cuda")
            # SDPA 直接使用原始连续 K/V，不经过分页 get；GQA 展开也在计时外。
            sdpa_key = key.repeat_interleave(4, dim=1).contiguous()
            sdpa_value = value.repeat_interleave(4, dim=1).contiguous()
            def sdpa_call():
                return torch.nn.functional.scaled_dot_product_attention(
                    query, sdpa_key, sdpa_value, dropout_p=0.0,
                    is_causal=False, scale=128 ** -0.5,
                )
            def cuda_call():
                return _C.paged_decode(query, storage._key, storage._value, table, length)
            expected, actual = sdpa_call(), cuda_call()
            error = (actual.float() - expected.float()).abs().max().item()
            print(f"[诊断] 长度={length}，计时外 CUDA / SDPA 最大绝对误差={error:.8g}，rtol=0.01，atol=0.002")
            assert actual.shape == expected.shape == query.shape
            assert actual.dtype == expected.dtype == torch.bfloat16
            assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
            torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.002)
            calls = (("SDPA", sdpa_call), ("CUDA V0", cuda_call))
            for _, call in calls:
                for _ in range(warmup):
                    call()
            samples = {name: [] for name, _ in calls}
            for run in range(repeats):
                # 交替先后顺序，减少总是让同一路径先测的影响，但不是严谨统计实验。
                for name, call in (calls if run % 2 == 0 else calls[::-1]):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    for _ in range(iterations):
                        call()
                    torch.cuda.synchronize()
                    per_call_us = (time.perf_counter() - started) * 1e6 / iterations
                    samples[name].append(per_call_us)
                    print(f"长度={length}，第 {run + 1} 组，{name}：平均调用耗时={per_call_us:.3f} us")
            for name, values in samples.items():
                print(f"[基线] 长度={length}，{name}：组平均调用耗时的中位数={statistics.median(values):.3f} us")
        finally:
            cache.release()
            assert storage._pool.num_free_blocks == length // 16
    print("[完成] 三种长度调用基线；不能据此宣称独立内核或端到端加速比")


@torch.inference_mode()
def check_int8() -> None:
    """同一 INT8 数据与 scale 的融合/独立反量化对照，不测试量化前后的模型质量。"""
    # 下一轮适配四个K scale后移除此暂停，不能用旧内核读取新布局。
    raise RuntimeError("INT8 CUDA对照暂时停用：K scale已改为四组，请先运行 python -m ampere_kv.paged_cache")
    generator = torch.Generator().manual_seed(2)
    # 同时覆盖空 warp、物理块边界、分段边界及两种头映射；不改变原 BF16 随机输入。
    for q_heads, kv_heads, length in ((2, 2, 1), (2, 2, 17), (2, 2, 65),
                                     (32, 8, 1), (32, 8, 17), (32, 8, 65)):
        blocks = max(3, (length + 15) // 16)
        storage = PagedKVStorage(kv_heads, 128, blocks, 16, device="cuda", kv_dtype=torch.int8)
        cache = PagedKVCache(storage)
        tensors = (storage._key, storage._value, storage._key_scale, storage._value_scale)
        for tensor in tensors:
            tensor.fill_(-128 if tensor.dtype == torch.int8 else float("nan"))
        held = [storage._pool.allocate() for _ in range(blocks)]
        for block in held:
            storage._pool.free(block)
        query = torch.randn(1, q_heads, 1, 128, generator=generator).to(torch.bfloat16)
        amplitude = torch.linspace(0.1, 2, kv_heads * length).reshape(1, kv_heads, length, 1)
        key = (torch.randn(1, kv_heads, length, 128, generator=generator) * amplitude).to(torch.bfloat16)
        value = (torch.randn(1, kv_heads, length, 128, generator=generator) * (3 - amplitude)).to(torch.bfloat16)
        # CPU 独立量化；不从被测分页 get 构造数学参考。FP16 scale 在 FP64 中精确读取。
        kd, ks = quantize_kv(key)
        vd, vs = quantize_kv(value)
        mapping = torch.arange(q_heads) // (q_heads // kv_heads)
        k = (kd.double() * ks.double()).index_select(1, mapping)
        v = (vd.double() * vs.double()).index_select(1, mapping)
        expected = torch.softmax((query.double() @ k.transpose(-2, -1)) * 128 ** -0.5, dim=-1) @ v
        try:
            cache.append(key.cuda(), value.cuda())
            table = torch.tensor(cache._table.block_ids, dtype=torch.long, device="cuda")
            q = query.cuda()
            addresses = tuple(t.data_ptr() for t in tensors)
            # 核对 GPU 写入确实对应同一组整数和 scale，避免把量化差异归咎于读取内核。
            for physical_tensor, reference in zip(tensors, (kd, vd, ks, vs)):
                physical = physical_tensor.cpu()
                for token in range(length):
                    block = cache._table.block_ids[token // 16]
                    assert torch.equal(physical[block, :, token % 16], reference[0, :, token])
            actual = _C.paged_decode_int8(q, *tensors, table, length)
            result = actual.cpu()
            error = (result.double() - expected).abs().max().item()
            print(f"[诊断] INT8 融合：Q头={q_heads}，KV头={kv_heads}，长度={length}，FP64参考误差={error:.8g}，rtol=0.01，atol=0.002")
            assert result.shape == query.shape and result.dtype == torch.bfloat16
            assert torch.isfinite(result).all() and torch.isfinite(expected).all()
            # 沿用 BF16 输出的初始容差；对照的是相同量化数据，而非原始 BF16 K/V。
            torch.testing.assert_close(result.double(), expected, rtol=0.01, atol=0.002)
            if length == 1:
                torch.testing.assert_close(result, v.to(torch.bfloat16), rtol=0, atol=0)
            bad_table = table.clone()
            bad_table[0] = blocks
            for bad_scale, selected_table in ((storage._key_scale.float(), table),
                                               (storage._key_scale[:, :, :, :0], table),
                                               (storage._key_scale, bad_table)):
                try:
                    _C.paged_decode_int8(q, storage._key, storage._value, bad_scale, storage._value_scale, selected_table, length)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("非法 scale 类型/形状或块编号未被拒绝")
            assert cache.length == length and tuple(t.data_ptr() for t in tensors) == addresses
        finally:
            cache.release()
        assert storage._pool.num_free_blocks == blocks
        print("[PASS] INT8 融合数值对照、非法输入拒绝和块归还通过")
    print("[PASS] INT8 CUDA 六个场景通过；未验证模型、长上下文、量化写入融合或性能")


def main() -> None:
    """默认仅检查当前 INT8 算子；保留 BF16 调用基线供后续比较。"""
    parser = argparse.ArgumentParser(description="INT8 分页 Decode 对照与 BF16 调用基线")
    parser.add_argument("--benchmark", action="store_true", help="测量 BF16 SDPA 与 CUDA 调用；不测 INT8")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("需要云端 CUDA GPU 与已编译的扩展")
    if args.benchmark:
        benchmark()
    else:
        check_int8()


if __name__ == "__main__":
    main()
