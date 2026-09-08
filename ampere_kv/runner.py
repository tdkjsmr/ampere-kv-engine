"""自建 Qwen3 路径：真实 Embedding、第一层 RMSNorm 与 Q/K/V 准备；暂不生成文本。"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from ampere_kv.reference import MODEL_ID, MODEL_REVISION


def rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """沿最后一维做 RMSNorm；按 Qwen3 的顺序保留归一化后的类型转换位置。"""

    # 每个 Token 单独计算均方值，不减均值；它不是 LayerNorm，也不是方差。
    # BF16 输入先转 FP32，降低平方和归约过程中的舍入误差。
    compute_states = hidden_states.float()
    mean_square = compute_states.square().mean(dim=-1, keepdim=True)
    normalized = compute_states * torch.rsqrt(mean_square + eps)
    # 先转回输入类型，再乘真实缩放权重；不能擅自换成乘完权重后才转 BF16。
    return normalized.to(hidden_states.dtype) * weight


def project_qkv(normalized: torch.Tensor, attention) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """使用真实权重投影并拆头，完成 Q/K 每头归一化；尚未应用 RoPE。"""

    # 显式使用权重做线性运算，不调用 HF Attention 的 forward；底层 GEMM 仍由 PyTorch 执行。
    query = torch.nn.functional.linear(normalized, attention.q_proj.weight, attention.q_proj.bias)
    key = torch.nn.functional.linear(normalized, attention.k_proj.weight, attention.k_proj.bias)
    value = torch.nn.functional.linear(normalized, attention.v_proj.weight, attention.v_proj.bias)
    # 投影宽度由真实权重决定；不能假设 Q/K/V 都与输入隐藏维度同宽。
    # [批大小, Token 数, 投影宽度] → [批大小, Token 数, 头数, 每头维度]。
    head_shape = (*normalized.shape[:-1], -1, attention.head_dim)
    query = query.reshape(head_shape)
    key = key.reshape(head_shape)
    value = value.reshape(head_shape)
    # 必须先拆头，再沿每头维度归一化；Q/K 使用各自的权重，V 不做 RMSNorm。
    query = rms_norm(query, attention.q_norm.weight, attention.q_norm.variance_epsilon)
    key = rms_norm(key, attention.k_norm.weight, attention.k_norm.variance_epsilon)
    # 返回缓存与 Attention 所需的 [批大小, 头数, Token 数, 每头维度]，不强制复制为连续张量。
    return query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)


def apply_rope(query: torch.Tensor, key: torch.Tensor, position_ids: torch.Tensor, config):
    """按固定 Qwen3 的默认 RoPE 旋转 Q/K；位置显式传入，不修改原张量或 V。"""

    # 只实现当前模型实际使用的完整维度默认 RoPE，不悄悄忽略长上下文缩放配置。
    if config.rope_scaling is not None or getattr(config, "partial_rotary_factor", 1.0) != 1.0:
        raise ValueError("当前 RoPE 仅支持无缩放、完整头维度的模型配置")
    dim = config.head_dim
    if dim % 2 != 0 or query.shape[-1] != dim or key.shape[-1] != dim:
        raise ValueError("RoPE 需要偶数头维度，且 Q/K 维度必须与配置一致")
    if position_ids.shape != (query.shape[0], query.shape[2]):
        raise ValueError("位置形状必须是 [批大小, 本次 Token 数]")
    # 从真实 rope_theta 计算各维度频率；与当前 CPU 加载后搬到 GPU 的模型路径一致。
    # 这里只生成很短的频率向量，不复制模型权重；后续接入多层时再复用频率表。
    exponent = torch.arange(0, dim, 2, dtype=torch.int64).float() / dim
    inv_freq = (1.0 / (config.rope_theta ** exponent)).to(query.device)
    with torch.autocast(device_type=query.device.type, enabled=False):
        # [批大小, 半个头维度, 1] @ [批大小, 1, Token 数]，得到各位置的旋转角度。
        frequencies = inv_freq[None, :, None].expand(query.shape[0], -1, 1)
        angles = (frequencies @ position_ids[:, None, :].to(device=query.device, dtype=torch.float32)).transpose(1, 2)
        # Qwen3 将前后两半配对，不是相邻偶/奇维度配对，因此角度表复制成两半。
        angles = torch.cat((angles, angles), dim=-1)
        cos = angles.cos().to(query.dtype).unsqueeze(1)
        sin = angles.sin().to(query.dtype).unsqueeze(1)
    half = dim // 2
    rotated_query = torch.cat((-query[..., half:], query[..., :half]), dim=-1)
    rotated_key = torch.cat((-key[..., half:], key[..., :half]), dim=-1)
    # 保持 HF 的逐项乘法再相加顺序，旋转后仍是 BF16；这里不涉及 Attention。
    return query * cos + rotated_query * sin, key * cos + rotated_key * sin


def main() -> None:
    """云端显式运行：复用一份 HF 权重，检查自建路径的第一段数值结果。"""

    if not torch.cuda.is_available():
        raise RuntimeError("真实权重对照需要在云端 CUDA 环境运行")
    text = input("请输入一段文本：")
    if not text.strip():
        raise ValueError("输入不能为空")
    # 复用已有模型版本常量，只读本地下载缓存，不保存用户输入。
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    formatted_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    input_ids = tokenizer(
        formatted_text, add_special_tokens=False, return_tensors="pt",
    )["input_ids"].to("cuda")

    # 暂时沿用已跑通的加载方式，不新写权重下载器或分片读取器。
    # 加载完整 BF16 模型，但只执行 Embedding 和第一层 Q/K/V 准备，不跑完整前向。
    print("正在从本地缓存加载固定版本 Qwen3-8B，用于真实权重对照……")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True,
    ).to("cuda")
    model.eval()

    with torch.inference_mode():
        embedding = model.model.embed_tokens
        first_norm = model.model.layers[0].input_layernorm
        # Embedding 是按 Token ID 查权重表的行，不是 one-hot 矩阵乘法。
        # 自建路径直接索引真实权重；参考路径调用 HF 持有的 Embedding 模块。
        hidden_states = embedding.weight[input_ids]
        reference_hidden = embedding(input_ids)
        torch.testing.assert_close(hidden_states, reference_hidden, rtol=0, atol=0)
        print(f"[PASS] Embedding 对照：形状={tuple(hidden_states.shape)}，类型={hidden_states.dtype}")

        # 使用第一层自己的缩放权重与 epsilon，而不是新建全 1 权重或硬编码 epsilon。
        normalized = rms_norm(hidden_states, first_norm.weight, first_norm.variance_epsilon)
        reference_normalized = first_norm(reference_hidden)
        max_error = (normalized.float() - reference_normalized.float()).abs().max().item()
        print(f"第一层 RMSNorm：形状={tuple(normalized.shape)}，类型={normalized.dtype}，最大绝对误差={max_error:.8g}")
        # 当前是相同原始运算和相同舍入顺序，先要求完全一致，不套用 Attention 容差。
        # 若后续换成融合内核，需另行确定容差，不能为了通过检查而自动放宽。
        torch.testing.assert_close(normalized, reference_normalized, rtol=0, atol=0)
        print("[PASS] 第一层 RMSNorm 真实权重对照通过；未验证完整模型或性能")

        attention = model.model.layers[0].self_attn
        # 先独立核对三组投影，出现差异时可区分 GEMM 与后面的拆头/归一化问题。
        for name, projection in (("Q", attention.q_proj), ("K", attention.k_proj), ("V", attention.v_proj)):
            projected = torch.nn.functional.linear(normalized, projection.weight, projection.bias)
            reference_projected = projection(reference_normalized)
            torch.testing.assert_close(projected, reference_projected, rtol=0, atol=0)
            print(f"[PASS] {name} 原始投影对照：形状={tuple(projected.shape)}，类型={projected.dtype}")

        query, key, value = project_qkv(normalized, attention)
        # 参考路径调用 HF 投影和 Q/K 归一化模块，明确使用配置中的头数拆分。
        batch, tokens, _ = reference_normalized.shape
        query_heads = model.config.num_attention_heads
        kv_heads = model.config.num_key_value_heads
        head_dim = model.config.head_dim
        reference_query = attention.q_norm(attention.q_proj(reference_normalized).reshape(batch, tokens, query_heads, head_dim)).transpose(1, 2)
        reference_key = attention.k_norm(attention.k_proj(reference_normalized).reshape(batch, tokens, kv_heads, head_dim)).transpose(1, 2)
        reference_value = attention.v_proj(reference_normalized).reshape(batch, tokens, kv_heads, head_dim).transpose(1, 2)
        for name, actual, expected in (("Q", query, reference_query), ("K", key, reference_key), ("V", value, reference_value)):
            max_error = (actual.float() - expected.float()).abs().max().item()
            print(f"{name} 拆头后对照：形状={tuple(actual.shape)}，类型={actual.dtype}，最大绝对误差={max_error:.8g}")
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert actual.dtype == torch.bfloat16
        assert query_heads % kv_heads == 0
        assert query.shape[1] == query_heads and key.shape[1] == value.shape[1] == kv_heads
        print(f"[PASS] Q/K/V 准备通过：Query 头={query_heads}，KV 头={kv_heads}，每组={query_heads // kv_heads}")
        # 整段位置从 0 开始；单 Token 复用最后一组真实 Q/K，但赋予下一位置 tokens。
        # 后者只验证非零位置的旋转，不宣称完成了下一个 Token 的模型前向。
        full_positions = torch.arange(tokens, device=query.device).unsqueeze(0)
        next_position = torch.tensor([[tokens]], device=query.device)
        for stage, q, k, positions in (
            ("整段 RoPE", query, key, full_positions),
            ("非零位置单 Token RoPE", query[:, :, -1:], key[:, :, -1:], next_position),
        ):
            actual_q, actual_k = apply_rope(q, k, positions, model.config)
            # 参考频率与旋转都由 HF 提供；同一份旋转前 Q/K 隔离本轮 RoPE 的误差。
            reference_cos, reference_sin = model.model.rotary_emb(q, positions)
            expected_q, expected_k = apply_rotary_pos_emb(q, k, reference_cos, reference_sin)
            for name, actual, expected in (("Q", actual_q, expected_q), ("K", actual_k, expected_k)):
                error = (actual.float() - expected.float()).abs().max().item()
                print(f"{stage} {name}：形状={tuple(actual.shape)}，类型={actual.dtype}，最大绝对误差={error:.8g}")
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            print(f"[PASS] {stage} 对照通过")
        # V 不传给旋转函数，再次确认它仍与未旋转的 HF V 一致。
        torch.testing.assert_close(value, reference_value, rtol=0, atol=0)
        print("[PASS] RoPE 后 V 保持不变；未写入 KV Cache，未验证完整 Attention 或模型生成")


if __name__ == "__main__":
    main()
