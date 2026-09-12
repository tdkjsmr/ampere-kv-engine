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
