"""CUDA Paged Decode V0 小型对照入口；不加载模型，也不测量性能。"""

import torch

from ampere_kv import _C
from ampere_kv.paged_cache import PagedKVCache, PagedKVStorage


def main() -> None:
    """检查直接分页读取、块边界、GQA 和独立数学参考；必须在云端 GPU 执行。"""
    if not torch.cuda.is_available():
        raise RuntimeError("此自检需要 CUDA GPU 和重新编译后的扩展")
    generator = torch.Generator().manual_seed(0)
    # 原有四个场景保持在前面，保留固定种子下原来的随机输入。
    cases = (("随机", 2, 2, 1), ("随机", 32, 8, 16), ("随机", 32, 8, 17),
             ("随机", 32, 8, 33), ("较长历史", 32, 8, 257),
             ("均匀权重", 32, 8, 33), ("大分数", 32, 8, 33))
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
    print("[PASS] CUDA V0 七个场景通过（最长 257 Token）；不代表完整长上下文、模型、性能或 CUDA Graph 验证通过")


if __name__ == "__main__":
    main()
