"""自建 Qwen3 路径：第一层完整 BF16 Prefill Decoder Layer 对照；暂不生成文本。"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ampere_kv.kv_cache import ContiguousKVCache, prefill_attention
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


@torch.no_grad()
def decoder_layer_prefill(
    hidden_states: torch.Tensor, layer, position_ids: torch.Tensor,
    cache: ContiguousKVCache, config,
) -> torch.Tensor:
    """用指定层的权重执行单请求、无填充 Prefill，返回整段隐藏状态。

    调用方提供该层专用的空缓存；函数写入 RoPE 后的 K 和未旋转的 V。
    不加载模型、不打印结果、不调用 HF 层的 forward，也不进行参考对照。
    不支持 Decode 或分块追加；缓存写入后若计算失败，不自动回滚。
    """
    attention = layer.self_attn
    # 在写入前拒绝不支持的配置，避免用普通 Attention 或 SiLU 静默代替其他结构。
    if attention.sliding_window is not None:
        raise ValueError("当前层 Prefill 不支持滑动窗口")
    if config.hidden_act != "silu":
        raise ValueError("当前 MLP 仅支持 SiLU 激活")
    if cache.length != 0:
        raise ValueError("层 Prefill 只接受空缓存，不支持分块追加")

    # Pre-Norm：先归一化，再送入 Attention；原始输入留在残差支路上。
    norm = layer.input_layernorm
    normalized = rms_norm(hidden_states, norm.weight, norm.variance_epsilon)
    query, key, value = project_qkv(normalized, attention)
    query, key = apply_rope(query, key, position_ids, config)
    head_output = prefill_attention(query, key, value, cache)
    # 合并所有 Query 头，再通过输出投影还原隐藏维度。
    batch, tokens, _ = hidden_states.shape
    merged = head_output.transpose(1, 2).contiguous().reshape(batch, tokens, -1)
    attention_output = torch.nn.functional.linear(merged, attention.o_proj.weight, attention.o_proj.bias)
    # 第一次残差加回本层原始输入；不原地修改输入，便于调用方保留或对照。
    after_attention = hidden_states + attention_output

    # HF 属性名表示 Attention 之后，但相对于 MLP，它仍是 Pre-Norm。
    mlp_norm = layer.post_attention_layernorm
    mlp_input = rms_norm(after_attention, mlp_norm.weight, mlp_norm.variance_epsilon)
    mlp = layer.mlp
    # gate 与 up 分别投影到中间维度；SiLU 只作用于 gate，然后逐元素相乘。
    gate = torch.nn.functional.linear(mlp_input, mlp.gate_proj.weight, mlp.gate_proj.bias)
    up = torch.nn.functional.linear(mlp_input, mlp.up_proj.weight, mlp.up_proj.bias)
    gated = torch.nn.functional.silu(gate) * up
    mlp_output = torch.nn.functional.linear(gated, mlp.down_proj.weight, mlp.down_proj.bias)
    # 第二次残差加回 Attention 残差后的状态，不是归一化结果或本层最初输入。
    return after_attention + mlp_output


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
    # 加载完整 BF16 模型，但只执行第一层 Decoder Layer，不跑完整模型前向。
    print("正在从本地缓存加载固定版本 Qwen3-8B，用于真实权重对照……")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True,
    ).to("cuda")
    model.eval()

    with torch.inference_mode():
        embedding = model.model.embed_tokens
        # Embedding 仍使用真实权重索引；HF 模块仅作为参考，不进入自建层函数。
        hidden_states = embedding.weight[input_ids]
        reference_hidden = embedding(input_ids)
        torch.testing.assert_close(hidden_states, reference_hidden, rtol=0, atol=0)
        print(f"[PASS] Embedding 对照：形状={tuple(hidden_states.shape)}，类型={hidden_states.dtype}")

        layer = model.model.layers[0]
        tokens = input_ids.shape[1]
        positions = torch.arange(tokens, device=hidden_states.device).unsqueeze(0)
        # 缓存由调用方创建；后续多层时每层各有一份，不能把不同层的 K/V 混用。
        cache = ContiguousKVCache(
            model.config.num_key_value_heads, model.config.head_dim,
            capacity=tokens + 1, device=hidden_states.device,
        )
        layer_output = decoder_layer_prefill(hidden_states, layer, positions, cache, model.config)
        assert cache.length == tokens
        stored_key, stored_value = cache.get()
        expected_shape = (1, model.config.num_key_value_heads, tokens, model.config.head_dim)
        assert stored_key.shape == stored_value.shape == expected_shape
        assert stored_key.dtype == stored_value.dtype == torch.bfloat16
        print(f"[PASS] 层函数 Prefill KV：有效长度={cache.length}，容量={cache.capacity}，形状={tuple(stored_key.shape)}")

        # HF 从原始 Embedding 独立执行整层；不读取自建缓存或任何自建中间结果。
        position_embeddings = model.model.rotary_emb(reference_hidden, positions)
        reference_layer_output = layer(
            hidden_states=reference_hidden, attention_mask=None,
            position_ids=positions, position_embeddings=position_embeddings,
            past_key_value=None, use_cache=False, output_attentions=False,
        )[0]
        error = (layer_output.float() - reference_layer_output.float()).abs().max().item()
        print(f"完整 Decoder Layer 输出：形状={tuple(layer_output.shape)}，类型={layer_output.dtype}，最大绝对误差={error:.8g}")
        assert layer_output.dtype == torch.bfloat16
        # 函数提取不应改变数值；仍保持第一层 BF16 严格对照，不放宽阈值。
        torch.testing.assert_close(layer_output, reference_layer_output, rtol=0, atol=0)
        print("[PASS] 可复用层函数的第一层 BF16 Prefill 对照通过；未验证真实 Decode、多层模型或性能")


if __name__ == "__main__":
    main()
