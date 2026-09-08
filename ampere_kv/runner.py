"""自建 Qwen3 路径：真实 Embedding、第一层 RMSNorm 与 Q/K/V 准备；暂不生成文本。"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
        print("本轮未应用 RoPE、未写入 KV Cache，也未验证完整 Attention 或模型生成")


if __name__ == "__main__":
    main()
