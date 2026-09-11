"""每 Token、每 KV 头的对称 INT8 量化参考；不接入缓存、模型或 CUDA 扩展。"""

import torch


@torch.no_grad()
def quantize_kv(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """输入 BF16 [1, KV头, Token数, 维度]，返回 INT8 数据与 FP16 [..., 1] scale。

    K、V 分别调用，不共享 scale。沿最后一维求最大值，不混合不同头或 Token。
    这是清晰优先的参考：有限性检查可能同步 GPU，不用于性能计时。
    """
    if tensor.ndim != 4 or tensor.shape[0] != 1 or any(size <= 0 for size in tensor.shape):
        raise ValueError("输入必须是非空的 [1, KV头数, Token数, 每头维度]")
    if tensor.dtype != torch.bfloat16:
        raise ValueError("量化输入必须是 BF16")
    if not torch.isfinite(tensor).all().item():
        raise ValueError("量化输入不能含 NaN 或 Inf")

    values = tensor.float()
    maximum = values.abs().amax(dim=-1, keepdim=True)
    # 全零向量固定 scale=1；非零向量最小使用 FP16 最小正规数 2^-14。
    # 不依赖 FP16 次正规数，避免跨实现下溢差异；极小值会因此损失更多精度。
    candidate = (maximum / 127.0).clamp_min(torch.finfo(torch.float16).tiny)
    candidate = torch.where(maximum == 0, torch.ones_like(candidate), candidate)
    # 不默默截断过大的 scale：该参考明确拒绝 FP16 正常范围无法承载的输入。
    if (candidate > torch.finfo(torch.float16).max).any().item():
        raise ValueError("量化 scale 超过 FP16 有限范围")
    scale = candidate.to(torch.float16)
    # 用实际保存的 scale 量化；后续反量化读取同一值，避免两端缩放约定不一致。
    # torch.round 的半整数向最近偶数舍入；不使用 -128，保持范围对称。
    data = torch.round(values / scale.float()).clamp(-127, 127).to(torch.int8)
    return data, scale


@torch.no_grad()
def dequantize_kv(data: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """用保存的 FP16 scale 还原为 FP32 独立张量；仅作数值参考，不是融合路径。"""
    if data.ndim != 4 or data.shape[0] != 1 or any(size <= 0 for size in data.shape):
        raise ValueError("INT8 数据必须是非空的 [1, KV头数, Token数, 每头维度]")
    if data.dtype != torch.int8 or scale.dtype != torch.float16:
        raise ValueError("反量化要求 INT8 数据与 FP16 scale")
    if scale.shape != data.shape[:-1] + (1,) or scale.device != data.device:
        raise ValueError("scale 必须与数据同设备，且形状为 [1, KV头数, Token数, 1]")
    if not (torch.isfinite(scale) & (scale > 0)).all().item():
        raise ValueError("scale 必须有限且为正数")
    if (data == -128).any().item():
        raise ValueError("对称量化数据不能含 -128")
    return data.float() * scale.float()


def main() -> None:
    """小型 CPU 自检：验证数值规则与字节数，不代表 Attention 或模型质量通过。"""
    generator = torch.Generator().manual_seed(0)
    values = torch.randn(1, 2, 3, 128, generator=generator)
    # 在同一小张量中覆盖全零、普通正负数、不同幅度、离群值和极小非零值。
    values[0, 0, 0].zero_()
    values[0, 0, 1] *= 0.01
    values[0, 1, 0] *= 10.0
    values[0, 1, 1, 0] = 100.0
    values[0, 1, 2] *= 1e-8
    original = values.to(torch.bfloat16)
    data, scale = quantize_kv(original)
    restored = dequantize_kv(data, scale)
    assert data.shape == original.shape and restored.dtype == torch.float32
    assert scale.shape == (1, 2, 3, 1)
    assert scale[0, 0, 0, 0].item() == 1.0 and not data[0, 0, 0].any().item()
    assert scale[0, 1, 2, 0].item() == torch.finfo(torch.float16).tiny
    # 逐向量单独调用应与批量处理一致，防止错误地跨 Token 或头计算最大值。
    for head in range(2):
        for token in range(3):
            part = original[:, head:head + 1, token:token + 1, :]
            part_data, part_scale = quantize_kv(part)
            assert torch.equal(part_data, data[:, head:head + 1, token:token + 1, :])
            assert torch.equal(part_scale, scale[:, head:head + 1, token:token + 1, :])
    # 舍入误差不超过半步；scale 向下取整可能造成边界截断，需另计超出范围部分。
    error = (restored - original.float()).abs()
    maximum = original.float().abs().amax(dim=-1, keepdim=True)
    bound = torch.maximum(scale.float() / 2, (maximum - 127 * scale.float()).clamp_min(0))
    allowance = 8 * torch.finfo(torch.float32).eps * maximum.clamp_min(1)
    assert torch.isfinite(restored).all().item() and (error <= bound + allowance).all().item()
    print(f"[PASS] 量化范围、独立 scale、零向量、极小值及舍入/截断误差界通过；最大绝对误差={error.max().item():.8g}")
    # scale 恰好为 1，直接检验正负半整数的最近偶数舍入规则。
    ties = torch.tensor([127, -127, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5], dtype=torch.bfloat16).reshape(1, 1, 1, 8)
    tie_data, _ = quantize_kv(ties)
    assert tie_data.flatten().tolist() == [127, -127, 0, 2, 2, 0, -2, -2]
    print("[PASS] 半整数最近偶数舍入规则通过")
    for bad in (original.float(), torch.full_like(original, float("nan")), torch.full_like(original, 1e10)):
        try:
            quantize_kv(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("非法类型、非有限值或 scale 溢出未被拒绝")
    before = original.numel() * original.element_size()
    after = data.numel() * data.element_size() + scale.numel() * scale.element_size()
    assert before == 1536 and after == 780
    print(f"[PASS] 非法输入拒绝；BF16={before} 字节，INT8+scale={after} 字节，理论容量比={before / after:.6f}")
    print("[PASS] CPU INT8 量化参考自检通过；未验证分页存储、Attention、模型、GPU 或性能")


if __name__ == "__main__":
    main()
