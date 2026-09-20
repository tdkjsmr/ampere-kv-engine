"""CUDA Paged Decode V0 对照与可选调用基线；不加载模型。"""

import argparse
import statistics
import subprocess
import sys
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
def check_bf16_write() -> None:
    """BF16 批量分页写入的逐元素对照；只验证搬运，不含量化或 Attention。"""
    generator = torch.Generator().manual_seed(4)
    # 两头一次整段写覆盖空缓存 Prefill 与跨块；八头分三批覆盖非对齐批量追加和跨块单 Token。
    for kv_heads, ends in ((2, (17,)), (8, (15, 32, 33))):
        length = ends[-1]
        blocks = max(3, (length + 15) // 16)
        storage = PagedKVStorage(kv_heads, 128, blocks, 16, device="cuda")
        cache = PagedKVCache(storage)
        tensors = (storage._key, storage._value)
        for tensor in tensors:
            tensor.fill_(float("nan"))
        # 倒序归还块号，使物理编号顺序不同于逻辑顺序。
        held = [storage._pool.allocate() for _ in range(blocks)]
        for block in held:
            storage._pool.free(block)
        key = torch.randn(1, kv_heads, length, 128, generator=generator).to(device="cuda", dtype=torch.bfloat16)
        value = torch.randn(1, kv_heads, length, 128, generator=generator).to(device="cuda", dtype=torch.bfloat16)
        try:
            start = 0
            for end in ends:
                table = cache.append(key[:, :, start:end], value[:, :, start:end], fused=True)
                assert cache.length == end and table.tolist() == list(cache._table.block_ids)
                start = end
            # 一次性取回再比对，避免逐 Token 同步；写入值必须与输入逐位相同。
            host = [tensor.cpu() for tensor in tensors]
            host_input = (key.cpu(), value.cpu())
            written = torch.zeros(blocks, 16, dtype=torch.bool)
            for token in range(length):
                block = cache._table.block_ids[token // 16]
                written[block, token % 16] = True
                for physical, reference in zip(host, host_input):
                    assert torch.equal(physical[block, :, token % 16], reference[0, :, token])
            # 未分配块与末块未写尾部仍是哨兵，证明写入不越界、不覆盖已有历史。
            for physical in host:
                assert torch.isnan(physical.permute(0, 2, 1, 3)[~written]).all()
        finally:
            cache.release()
        assert storage._pool.num_free_blocks == blocks
        print(f"[PASS] BF16 批量写入：KV头={kv_heads}，分批={ends}，逐元素一致且块已归还")
    print("[PASS] BF16分页写入通过：整段Prefill、非对齐批量追加、跨块单Token、非连续物理块；未验证模型或性能")


@torch.inference_mode()
def check_int8() -> None:
    """同一 INT8 数据与 scale 的融合/独立反量化对照，不测试量化前后的模型质量。"""
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
        # 首Token覆盖零scale特例、极小值和最近偶数舍入；其他Token保持随机分布。
        key[:, :, 0, :32] = 0
        key[:, :, 0, 32:64] = 1e-8
        key[:, :, 0, 64:96] = 0
        key[:, :, 0, 64:69] = torch.tensor([127, 0.5, 1.5, -0.5, -1.5], dtype=torch.bfloat16)
        value[:, :, 0] = 0
        value[:, :, 0, :5] = torch.tensor([127, 0.5, 1.5, -0.5, -1.5], dtype=torch.bfloat16)
        # CPU 独立量化；不从被测分页 get 构造数学参考。FP16 scale 在 FP64 中精确读取。
        # 按切片独立构造四组参考，与存储内部的reshape写法分开。
        groups = [quantize_kv(key[..., start:start + 32]) for start in range(0, 128, 32)]
        kd = torch.cat([data for data, scale in groups], dim=-1)
        ks = torch.cat([scale for data, scale in groups], dim=-1)
        vd, vs = quantize_kv(value)
        mapping = torch.arange(q_heads) // (q_heads // kv_heads)
        k = torch.cat([data.double() * scale.double() for data, scale in groups], dim=-1).index_select(1, mapping)
        v = (vd.double() * vs.double()).index_select(1, mapping)
        expected = torch.softmax((query.double() @ k.transpose(-2, -1)) * 128 ** -0.5, dim=-1) @ v
        try:
            gpu_key, gpu_value = key.cuda(), value.cuda()
            # 两头场景整段写；GQA场景从非对齐位置批量追加，最后再写一个Decode Token。
            ends = [length] if q_heads == 2 or length == 1 else [15, length - 1, length]
            start = 0
            for end in ends:
                if end > start:
                    table = cache.append(gpu_key[:, :, start:end], gpu_value[:, :, start:end], fused=True)
                    start = end
            q = query.cuda()
            # 核对 GPU 写入确实对应同一组整数和 scale，避免把量化差异归咎于读取内核。
            for physical_tensor, reference in zip(tensors, (kd, vd, ks, vs)):
                physical = physical_tensor.cpu()
                written = torch.zeros(blocks, 16, dtype=torch.bool)
                for token in range(length):
                    block = cache._table.block_ids[token // 16]
                    written[block, token % 16] = True
                    assert torch.equal(physical[block, :, token % 16], reference[0, :, token])
                # 未分配块和尾部仍是哨兵，验证写入不越界、不误覆盖。
                untouched = physical.permute(0, 2, 1, 3)[~written]
                assert (untouched == -128).all() if physical.dtype == torch.int8 else torch.isnan(untouched).all()
            actual = _C.paged_decode_int8(q, *tensors, table, length)
            result = actual.cpu()
            error = (result.double() - expected).abs().max().item()
            print(f"[诊断] INT8 融合：Q头={q_heads}，KV头={kv_heads}，长度={length}，FP64参考误差={error:.8g}，rtol=0.01，atol=0.002")
            # 沿用 BF16 输出的初始容差；对照的是相同量化数据，而非原始 BF16 K/V。
            torch.testing.assert_close(result.double(), expected, rtol=0.01, atol=0.002)
            if length == 1:
                torch.testing.assert_close(result, v.to(torch.bfloat16), rtol=0, atol=0)
            bad_table = table.clone()
            bad_table[0] = blocks
            try:
                _C.paged_decode_int8(q, *tensors, bad_table, length)
            except RuntimeError:
                pass
            else:
                raise AssertionError("非法块编号未被拒绝")
            assert cache.length == length
        finally:
            cache.release()
        assert storage._pool.num_free_blocks == blocks
        print("[PASS] INT8 融合数值对照、非法输入拒绝和块归还通过")
    print("[PASS] INT8批量Prefill写入与Decode追加六个场景通过；未验证模型、长上下文或性能")


@torch.inference_mode()
def probe_internal_guard(kv_dtype: str) -> None:
    """独立子进程专用：内部入口遇非法块号必须设备端失败，不得越界访问或正常返回。

    设备端断言会毒化整个 CUDA 上下文，之后任何 CUDA 调用都失败，所以本函数只能在独占
    进程里运行，正常检查进程不得调用。执行到最后一行打印就说明保护没有生效。
    """
    quantized = kv_dtype == "int8"
    storage = PagedKVStorage(2, 128, 3, 16, device="cuda",
                             kv_dtype=torch.int8 if quantized else torch.bfloat16)
    cache = PagedKVCache(storage)
    key = torch.randn(1, 2, 16, 128, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(1, 2, 16, 128, device="cuda", dtype=torch.bfloat16)
    try:
        table = cache.append(key, value, fused=True)
        query = torch.randn(1, 2, 1, 128, device="cuda", dtype=torch.bfloat16)
        bad_table = table.clone()
        bad_table[0] = 3  # 等于物理块总数，越界一个块
        print(f"[探测开始] {kv_dtype} 内部入口非法块号", flush=True)
        if quantized:
            _C.paged_decode_int8_internal(query, storage._key, storage._value,
                                          storage._key_scale, storage._value_scale, bad_table, 16)
        else:
            _C.paged_decode_internal(query, storage._key, storage._value, bad_table, 16)
        torch.cuda.synchronize()
    finally:
        cache.release()
    print("[未拦截] 内部入口对非法块号正常返回，设备端保护未生效")


def check_internal_guard() -> None:
    """在独立子进程中确认内部入口的设备端块号保护在实际构建配置下真的生效。

    设备端断言与非法访存的错误信息不同：用它区分"保护拦住了"和"保护没编进去、真的读
    越界了"。不匹配具体断言文本，避免绑定某个 CUDA/PyTorch 版本的措辞。
    """
    for kv_dtype in ("bf16", "int8"):
        result = subprocess.run(
            [sys.executable, "-m", "ampere_kv.paged_decode", "--probe-guard", kv_dtype],
            capture_output=True, text=True,
        )
        # 先确认子进程真的走到了探测点，否则"失败"可能只是导入或建存储出错。
        assert f"[探测开始] {kv_dtype}" in result.stdout, (
            f"{kv_dtype}：子进程未到达探测点，无法判定保护是否生效；stderr 末尾={result.stderr[-300:]}")
        assert "[未拦截]" not in result.stdout, f"{kv_dtype}：内部入口未拦截非法块号，进程正常返回"
        assert result.returncode != 0, f"{kv_dtype}：探测进程没有失败，设备端保护未生效"
        assert "illegal memory access" not in result.stderr, (
            f"{kv_dtype}：失败原因是非法访存而非设备端断言，说明保护未在访存前生效；"
            f"stderr 末尾={result.stderr[-300:]}")
        print(f"[PASS] {kv_dtype} 内部入口的设备端块号保护已在独立子进程确认生效（异步失败，运行需终止）")


def main() -> None:
    """默认检查 BF16 写入、INT8 算子与内部入口保护；保留 BF16 调用基线供后续比较。"""
    parser = argparse.ArgumentParser(description="分页写入、INT8 Decode 对照与内部入口保护检查、BF16 调用基线")
    parser.add_argument("--benchmark", action="store_true", help="测量 BF16 SDPA 与 CUDA 调用；不测 INT8")
    parser.add_argument("--probe-guard", choices=("bf16", "int8"),
                        help="独立子进程专用：探测内部入口的设备端块号保护，预期失败退出")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("需要云端 CUDA GPU 与已编译的扩展")
    if args.probe_guard:
        probe_internal_guard(args.probe_guard)
        return
    if args.benchmark:
        benchmark()
    else:
        check_bf16_write()
        check_int8()
        check_internal_guard()


if __name__ == "__main__":
    main()
