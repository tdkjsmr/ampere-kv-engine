"""自建 Qwen3 路径：BF16 Prefill、末尾归一化与首个贪心 Token 对照；暂不执行 Decode。"""

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
    """云端显式运行：复用一份 HF 权重，检查自建多层 Prefill 的隐藏状态。"""

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
    # 加载一份 BF16 权重，执行完整 Prefill 并选出首个 Token；暂不执行生成循环。
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

        tokens = input_ids.shape[1]
        positions = torch.arange(tokens, device=hidden_states.device).unsqueeze(0)
        layers = model.model.layers
        assert len(layers) == model.config.num_hidden_layers
        # 当前各层使用相同位置；HF 参考频率只生成一次，不依赖自建路径的中间结果。
        position_embeddings = model.model.rotary_emb(reference_hidden, positions)
        # 列表下标对应层号；保留全部缓存，不能反复覆盖或复用第一层的存储。
        caches = []
        expected_shape = (1, model.config.num_key_value_heads, tokens, model.config.head_dim)
        for layer_index, layer in enumerate(layers):
            cache = ContiguousKVCache(
                model.config.num_key_value_heads, model.config.head_dim,
                capacity=tokens + 1, device=hidden_states.device,
            )
            caches.append(cache)
            # 两条路径各自接收上一层输出；不能每层用 HF 状态重置自建输入。
            hidden_states = decoder_layer_prefill(hidden_states, layer, positions, cache, model.config)
            reference_hidden = layer(
                hidden_states=reference_hidden, attention_mask=None,
                position_ids=positions, position_embeddings=position_embeddings,
                past_key_value=None, use_cache=False, output_attentions=False,
            )[0]
            # 每层都检查缓存与隐藏状态；出现首个差异立即停止，不放宽容差。
            assert cache.length == tokens
            stored_key, stored_value = cache.get()
            assert stored_key.shape == stored_value.shape == expected_shape
            assert stored_key.dtype == stored_value.dtype == torch.bfloat16
            assert hidden_states.dtype == torch.bfloat16
            error = (hidden_states.float() - reference_hidden.float()).abs().max().item()
            print(f"第 {layer_index} 层：输出形状={tuple(hidden_states.shape)}，最大绝对误差={error:.8g}，KV 长度={cache.length}，容量={cache.capacity}")
            torch.testing.assert_close(hidden_states, reference_hidden, rtol=0, atol=0)
            print(f"[PASS] 第 {layer_index} 层 BF16 Prefill 严格对照通过")

        # 循环结束后仍保留每层自己的缓存，并确认后续层没有改变前面层的有效长度。
        assert len(caches) == len(layers) and all(cache.length == tokens for cache in caches)
        # 所有缓存同时存活，K/V 起始地址必须各不相同，排除意外共享存储。
        addresses = [tensor.data_ptr() for cache in caches for tensor in cache.get()]
        assert len(set(addresses)) == 2 * len(layers)
        print(f"[PASS] 全部 {len(layers)} 层 BF16 Prefill 与独立 KV 缓存检查通过")

        # Final Norm 使用模型末尾独立的权重，不是最后一层内部的任意一个 RMSNorm。
        final_norm = model.model.norm
        normalized = rms_norm(hidden_states, final_norm.weight, final_norm.variance_epsilon)
        reference_normalized = final_norm(reference_hidden)
        error = (normalized.float() - reference_normalized.float()).abs().max().item()
        print(f"Final RMSNorm：形状={tuple(normalized.shape)}，类型={normalized.dtype}，最大绝对误差={error:.8g}")
        assert normalized.dtype == torch.bfloat16
        torch.testing.assert_close(normalized, reference_normalized, rtol=0, atol=0)
        print("[PASS] Final RMSNorm 严格对照通过")

        # 只有最后一个输入位置预测首个输出 Token；保留长度为 1 的序列维度。
        # LM Head 将隐藏维度映射到词表大小，使用自己的权重，不假定与 Embedding 共享。
        logits = torch.nn.functional.linear(normalized[:, -1:, :], model.lm_head.weight, model.lm_head.bias)
        # 再由 HF 完整 forward 从原始 Token IDs 独立计算，覆盖模型入口到词表输出。
        # 两边都只投影最后一个位置，避免因 GEMM 形状不同引入额外比较差异。
        # 不创建第二份模型，不传入我们的缓存，也不让 HF 保存参考 KV。
        reference_logits = model(
            input_ids=input_ids, position_ids=positions, use_cache=False,
            logits_to_keep=1, return_dict=True,
        ).logits
        assert logits.shape == (1, 1, model.config.vocab_size)
        assert logits.dtype == torch.bfloat16
        error = (logits.float() - reference_logits.float()).abs().max().item()
        print(f"首个 Token logits：形状={tuple(logits.shape)}，类型={logits.dtype}，最大绝对误差={error:.8g}")
        torch.testing.assert_close(logits, reference_logits, rtol=0, atol=0)
        print("[PASS] 自建 Prefill logits 与 HF 完整 forward 严格对照通过")

        # 贪心选择直接取最大分数的索引，不需要 Softmax，也不使用采样配置。
        next_token_id = logits[0, 0].argmax().item()
        reference_token_id = reference_logits[0, 0].argmax().item()
        print(f"first_token_id = {next_token_id}，reference_token_id = {reference_token_id}")
        assert next_token_id == reference_token_id
        # Token 可能只对应部分字符；显示仅供观察，正确性以 ID 和 logits 为准。
        print(f"first_token_text = {tokenizer.decode([next_token_id], skip_special_tokens=True)!r}")
        # 首个 Token 尚未送回模型，因此所有缓存仍只包含原始输入的 K/V。
        assert all(cache.length == tokens for cache in caches)
        print("[PASS] 自建 BF16 Prefill 首个贪心 Token 对照通过；未执行真实 Decode、连续生成或性能测量")


if __name__ == "__main__":
    main()
