"""CUDA Paged Decode V0 对照与可选调用基线；不加载模型。"""

import argparse
import statistics
import time

import torch

from ampere_kv import _C
from ampere_kv.paged_cache import PagedKVCache, PagedKVStorage


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


def main() -> None:
    """检查直接分页读取、块边界、GQA 和独立数学参考；必须在云端 GPU 执行。"""
    parser = argparse.ArgumentParser(description="CUDA Paged Decode 对照与调用基线")
    parser.add_argument("--benchmark", action="store_true", help="测量预先准备好数据的 SDPA 与 CUDA V0 调用；默认运行数值与分段边界对照")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("此自检需要 CUDA GPU 和重新编译后的扩展")
    if args.benchmark:
        benchmark()
        return
    generator = torch.Generator().manual_seed(0)
    # 原有四个场景保持在前面，保留固定种子下原来的随机输入。
    cases = (("随机", 2, 2, 1), ("随机", 32, 8, 16), ("随机", 32, 8, 17),
             ("随机", 32, 8, 33), ("较长历史", 32, 8, 257),
             ("均匀权重", 32, 8, 33), ("大分数", 32, 8, 33),
             # 追加场景不改变前七项随机输入；覆盖单段末端、整段和不足一段的尾部。
             ("分段边界", 32, 8, 256), ("分段边界", 32, 8, 512),
             ("分段边界", 32, 8, 513), ("均匀权重", 32, 8, 513),
             ("大分数", 32, 8, 513),
             # 放在末尾，保留前十二项随机输入；检查新段长的单段末端及第二段首个 Token。
             ("分段边界", 32, 8, 64), ("分段边界", 32, 8, 65))
    for label, q_heads, kv_heads, length in cases:
        num_blocks = max(3, (length + 15) // 16)
        storage = PagedKVStorage(kv_heads, 128, num_blocks, 16, device="cuda")
        cache = PagedKVCache(storage)
        # 空闲数据故意设为 NaN，若读过有效尾部，输出的有限值检查应失败。
        storage._key.fill_(float("nan"))
        storage._value.fill_(float("nan"))
        held = [storage._pool.allocate() for _ in range(num_blocks)]
        # 原有三块顺序仍为 (2, 0, 1)；更长场景继续使用非逻辑顺序的物理块。
        for block in (held[1], held[0], *held[2:]):
            storage._pool.free(block)
        query = torch.randn(1, q_heads, 1, 128, generator=generator).to(torch.bfloat16)
        key = torch.randn(1, kv_heads, length, 128, generator=generator).to(torch.bfloat16)
        value = torch.randn(1, kv_heads, length, 128, generator=generator).to(torch.bfloat16)
        if label == "均匀权重":
            query.zero_()  # 全部分数为零，输出应为有效 V 的均值。
        elif label == "大分数":
            query.fill_(8)
            # 分数从约 -724 递增到 +724，反复更新在线最大值。
            # 直接对正分数取 FP32 exp 会溢出；稳定 softmax 应仍保持有限。
            ramp = torch.linspace(-8, 8, length).to(torch.bfloat16)
            key.copy_(ramp.reshape(1, 1, length, 1).expand_as(key))
        print(f"[场景] {label}：Q头={q_heads}，KV头={kv_heads}，长度={length}，rtol=0.01，atol=0.002")
        cache.append(key.cuda(), value.cuda())  # 长度已包含当前 Token，内核不能再追加。
        table = torch.tensor(cache._table.block_ids, dtype=torch.long, device="cuda")
        actual = _C.paged_decode(query.cuda(), storage._key, storage._value, table, length)
        # 现有分页读回 + SDPA 路径：只计算，不再调用会重复追加的 Decode 包装。
        cached_key, cached_value = cache.get()
        group = q_heads // kv_heads
        sdpa = torch.nn.functional.scaled_dot_product_attention(
            query.cuda(), cached_key.repeat_interleave(group, dim=1),
            cached_value.repeat_interleave(group, dim=1), dropout_p=0.0,
            is_causal=False, scale=128 ** -0.5,
        )
        # 再用 CPU FP64 的显式公式独立核对，不受 GPU SDPA 后端选择影响。
        mapping = torch.arange(q_heads) // group
        k = key.double().index_select(1, mapping)
        v = value.double().index_select(1, mapping)
        scores = (query.double() @ k.transpose(-2, -1)) * 128 ** -0.5
        expected = torch.softmax(scores, dim=-1) @ v
        torch.cuda.synchronize()
        result = actual.cpu()
        assert result.shape == query.shape and result.dtype == torch.bfloat16
        sdpa_result = sdpa.cpu()
        error = (result.double() - expected).abs().max().item()
        sdpa_error = (result.float() - sdpa_result.float()).abs().max().item()
        # 诊断放在断言前，失败时也能保留场景、分数范围与两条对照误差。
        print(f"[诊断] 分数范围=[{scores.min().item():.6g}, {scores.max().item():.6g}]，FP64参考误差={error:.8g}，SDPA误差={sdpa_error:.8g}")
        assert torch.isfinite(result).all() and torch.isfinite(sdpa_result).all() and torch.isfinite(expected).all()
        # 沿用原有初始容差，允许 BF16 输出舍入与不同归约顺序，不因新增场景放宽。
        # 不是模型或 INT8 精度标准；失败时先诊断，不能为了通过直接放宽。
        torch.testing.assert_close(result.double(), expected, rtol=0.01, atol=0.002)
        torch.testing.assert_close(result, sdpa_result, rtol=0.01, atol=0.002)
        if label == "均匀权重":
            mean_value = v.mean(dim=2, keepdim=True)
            torch.testing.assert_close(expected, mean_value, rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(result.double(), mean_value, rtol=0.01, atol=0.002)
        elif label == "大分数":
            assert scores.min().item() < -100 and scores.max().item() > 100
        if length == 1:
            torch.testing.assert_close(result, value, rtol=0, atol=0)
        # 非法物理编号必须在启动读取内核前拒绝，不能让 GPU 越界访存。
        bad_table = table.clone()
        bad_table[0] = storage._key.shape[0]
        try:
            _C.paged_decode(query.cuda(), storage._key, storage._value, bad_table, length)
        except RuntimeError:
            pass
        else:
            raise AssertionError("非法物理块编号未被拒绝")
        cache.release()
        assert storage._pool.num_free_blocks == num_blocks
        print(f"[PASS] {label}：数值对照、非法块号拒绝与块归还通过")
    print(f"[PASS] CUDA 分页 Decode {len(cases)} 个场景通过（最长 513 Token）；不代表完整长上下文、模型、性能或 CUDA Graph 验证通过")


if __name__ == "__main__":
    main()
