"""自建 Qwen3 BF16 生成、HF 逐步对照与单请求性能基线。"""

import argparse
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ampere_kv.kv_cache import ContiguousKVCache, prefill_attention, decode_attention
from ampere_kv.paged_cache import PagedKVCache, PagedKVStorage
# 模型与分词器共用固定版本；旧参考演示由本文件的 check 模式取代。
MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"

# 本轮最多生成 32 个新 Token，包含 Prefill 选出的第一个；遇到 EOS 提前结束。
MAX_NEW_TOKENS = 32
# 分页参考先固定块大小，不在本轮引入调优参数。
PAGED_BLOCK_SIZE = 16


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
def decoder_layer_forward(
    hidden_states: torch.Tensor, layer, position_ids: torch.Tensor,
    cache: ContiguousKVCache | PagedKVCache, config, *, is_prefill: bool, cuda_decode: bool = False,
) -> torch.Tensor:
    """执行单请求、无填充的整段 Prefill 或单 Token Decode，返回本次隐藏状态。

    调用方提供该层专用缓存并显式选择阶段；写入 RoPE 后的 K 和未旋转的 V。
    不加载模型、不打印结果、不调用 HF 层的 forward，也不进行参考对照。
    不支持分块 Prefill；缓存写入后若计算失败，不自动回滚。
    cuda_decode 用于可选生成与对照，默认仍走 SDPA；当前 CUDA 分支不支持 Graph。
    """
    attention = layer.self_attn
    if cuda_decode and (is_prefill or not isinstance(cache, PagedKVCache)):
        raise ValueError("CUDA Decode 只接受分页缓存的单 Token Decode")
    if is_prefill and cache.length != 0:
        raise ValueError("层 Prefill 只接受空缓存，不支持分块追加")
    if not is_prefill:
        if hidden_states.shape[1] != 1 or cache.length == 0:
            raise ValueError("层 Decode 需要单 Token 输入和非空历史缓存")

    # Pre-Norm：先归一化，再送入 Attention；原始输入留在残差支路上。
    norm = layer.input_layernorm
    normalized = rms_norm(hidden_states, norm.weight, norm.variance_epsilon)
    query, key, value = project_qkv(normalized, attention)
    query, key = apply_rope(query, key, position_ids, config)
    # 只有缓存 Attention 的阶段不同，归一化、投影、RoPE、残差与 MLP 共用原实现。
    # CUDA 分页且块大小16、每头128维时，两种精度的 Prefill 共用批量写入；其余仍走参考实现。
    fused_prefill = (is_prefill and isinstance(cache, PagedKVCache) and cache._key.is_cuda
                     and cache._key.shape[2] == 16 and cache._key.shape[3] == 128)
    if fused_prefill:
        # 先批量写缓存，再用原始 BF16 K/V 做 SDPA；BF16 也不从分页存储读回历史。
        # 写入是逐位搬运（INT8 为量化），Attention 输入、GQA 映射、掩码和 scale 都不变。
        cache.append(key, value, fused=True)
        group_size = query.shape[1] // key.shape[1]
        head_output = torch.nn.functional.scaled_dot_product_attention(
            query.contiguous(), key.repeat_interleave(group_size, dim=1).contiguous(),
            value.repeat_interleave(group_size, dim=1).contiguous(),
            dropout_p=0.0, is_causal=query.shape[2] > 1, scale=query.shape[-1] ** -0.5,
        )
    elif cuda_decode:
        # 按缓存类型选择Decode内核；融合写入在append内部按需导入同一扩展。
        from ampere_kv import _C

        # 单 Token K/V 只追加一次；两种精度都走批量写入入口并复用其返回的块表。
        # CUDA 直接读物理存储，不调用 get 或复制 GQA 头。
        table = cache.append(key, value, fused=True)
        if cache._key.dtype == torch.int8:
            head_output = _C.paged_decode_int8(
                query.contiguous(), cache._key, cache._value,
                cache._key_scale, cache._value_scale, table, cache.length,
            )
        else:
            head_output = _C.paged_decode(query.contiguous(), cache._key, cache._value, table, cache.length)
    else:
        attention_fn = prefill_attention if is_prefill else decode_attention
        head_output = attention_fn(query, key, value, cache)
    return finish_decoder_layer(hidden_states, head_output, layer)


def finish_decoder_layer(hidden_states: torch.Tensor, head_output: torch.Tensor, layer):
    """共用层后半段，返回完整层输出；不读写缓存，不进行对照。

    原始隐藏状态走残差支路，不能替换成归一化后的状态。
    """
    attention = layer.self_attn
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


@torch.no_grad()
def model_forward(model, input_ids, position_ids, caches, *, is_prefill: bool, cuda_decode: bool = False):
    """自建模型级前向：返回本次最后一个位置的 logits，并更新各层缓存。

    只读取 HF 对象持有的权重，不调用 HF 模型、层或输出头的 forward。
    不负责分词、选词、打印或对照；不提供缓存写入失败后的跨层回滚。
    """
    layers = model.model.layers
    if len(caches) != len(layers):
        raise ValueError("缓存数量必须与模型层数一致")
    if cuda_decode:
        # 全部层写入前拒绝不支持的后端配置；后续 GPU 故障仍不提供事务回滚。
        if is_prefill or model.config.head_dim != 128 or any(
            not isinstance(cache, PagedKVCache) or cache._key.shape[2] != 16 for cache in caches
        ):
            raise ValueError("CUDA V0 仅支持分页 Decode、块大小 16、每头 128 维")
    # 写入前检查全部层的长度和容量，避免后面某层才发现容量不足。
    used_tokens = caches[0].length
    if any(cache.length != used_tokens or used_tokens + input_ids.shape[1] > cache.capacity for cache in caches):
        raise ValueError("各层缓存长度不一致或容量不足")
    hidden_states = model.model.embed_tokens.weight[input_ids]
    for layer, cache in zip(layers, caches):
        hidden_states = decoder_layer_forward(
            hidden_states, layer, position_ids, cache, model.config, is_prefill=is_prefill, cuda_decode=cuda_decode,
        )
    # Prefill 和 Decode 共用末尾归一化与输出投影，不另写一套生成计算。
    return final_logits(model, hidden_states)


def final_logits(model, hidden_states):
    """共用末尾归一化和LM Head，只返回最后位置的logits，不读写缓存。"""
    norm = model.model.norm
    normalized = rms_norm(hidden_states, norm.weight, norm.variance_epsilon)
    return torch.nn.functional.linear(normalized[:, -1:, :], model.lm_head.weight, model.lm_head.bias)


@torch.inference_mode()
def generate_tokens(model, input_ids, *, max_new_tokens: int = MAX_NEW_TOKENS, verify: bool = False, timings: dict | None = None, cache_kind: str = "contiguous", cuda_decode: bool = False, kv_dtype: torch.dtype = torch.bfloat16, ignore_eos: bool = False) -> list[int]:
    """返回新 Token IDs；每次请求创建独立缓存，不向调用方保留 GPU 张量。

    两种模式共用自建前向和循环；verify 只额外执行 HF 参考和诊断。
    纯生成不调用 HF forward、不创建 HF 缓存、不逐步打印，仍保留必要输入检查。
    timings 非空时写入请求级墙钟指标；不允许同时开启 HF 对照。
    两种存储共用生成循环；分页每层独占块池，尚非多请求共享池。
    cuda_decode 仅切换 Decode，Prefill 始终使用 SDPA；默认后端不变。
    INT8支持分页CUDA生成与计时，不支持HF逐Token严格对照。
    ignore_eos仅用于固定工作量基线，普通生成仍遇到EOS停止。
    """
    if cache_kind not in ("contiguous", "paged"):
        raise ValueError("缓存类型必须是 contiguous 或 paged")
    if kv_dtype not in (torch.bfloat16, torch.int8):
        raise ValueError("KV类型必须是BF16或INT8")
    if kv_dtype == torch.int8 and (cache_kind != "paged" or not cuda_decode or verify):
        raise ValueError("INT8只支持分页CUDA生成，不支持HF严格对照")
    if cuda_decode and (cache_kind != "paged" or verify):
        raise ValueError("CUDA Decode 仅支持分页纯生成；逐步对照请使用 cuda-check")
    if timings is not None and verify:
        raise ValueError("计时不能包含 HF 对照，请使用纯生成路径")
    if max_new_tokens <= 0:
        raise ValueError("新 Token 上限必须为正数")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError("当前只支持单条非空、无填充的 Token 序列")
    if input_ids.dtype != torch.long or input_ids.device.type != "cuda":
        raise ValueError("输入 Token IDs 必须是 CUDA 上的 int64 张量")
    if model.training:
        raise ValueError("生成前必须将模型设为 eval 模式")
    tokens = input_ids.shape[1]
    if timings is not None:
        # 先排空前序 GPU 工作，再开始计时；模型加载、分词、输入搬运都已完成。
        torch.cuda.synchronize(input_ids.device)
        started = time.perf_counter()
    # 请求计时包含缓存分配，不是仅测某个 CUDA 内核。
    capacity = tokens + max_new_tokens
    caches = []
    for _ in model.model.layers:
        if cache_kind == "paged":
            # 整块向上取整；仅选择存储实现，不复制前向或生成循环。
            # Runner 仍是单请求：每层创建独立存储，再创建该请求的块表。
            cache = PagedKVCache(PagedKVStorage(
                model.config.num_key_value_heads, model.config.head_dim,
                num_blocks=(capacity + PAGED_BLOCK_SIZE - 1) // PAGED_BLOCK_SIZE,
                block_size=PAGED_BLOCK_SIZE, device=input_ids.device, kv_dtype=kv_dtype,
            ))
        else:
            cache = ContiguousKVCache(
                model.config.num_key_value_heads, model.config.head_dim,
                capacity=capacity, device=input_ids.device,
            )
        caches.append(cache)
    eos_ids = model.generation_config.eos_token_id
    eos_ids = [eos_ids] if isinstance(eos_ids, int) else (eos_ids or [])
    generated_ids = []
    used_tokens = 0
    current_input = input_ids
    if verify:
        # 参考状态只在对照模式创建；两条路径各自维护缓存和输出序列。
        reference_input, reference_cache, reference_ids = input_ids, None, []
        print(f"缓存={cache_kind}，预留长度={capacity}，每层物理容量={caches[0].capacity}")

    for step in range(max_new_tokens):
        # 首轮处理整段输入，后续只处理一个 Token；绝对位置从历史有效长度继续。
        positions = torch.arange(used_tokens, used_tokens + current_input.shape[1], device=input_ids.device).unsqueeze(0)
        logits = model_forward(model, current_input, positions, caches, is_prefill=(step == 0), cuda_decode=cuda_decode and step > 0)
        used_tokens += current_input.shape[1]
        next_id = logits[0, 0].argmax().item()
        if timings is not None:
            # 当前路径在同一流上计算；item() 已等待产生 Token 的 GPU 工作完成。
            # 在 CPU 可取得 Token ID 的时刻记时，不额外为每层/每 Token 插入同步。
            token_ready = time.perf_counter()
            if step == 0:
                first_ready = token_ready
        generated_ids.append(next_id)

        if verify:
            reference = model(
                input_ids=reference_input, position_ids=positions, cache_position=positions[0],
                past_key_values=reference_cache, use_cache=True, logits_to_keep=1, return_dict=True,
            )
            reference_cache = reference.past_key_values
            torch.testing.assert_close(logits, reference.logits, rtol=0, atol=0)
            reference_ids.append(reference.logits[0, 0].argmax().item())
            assert next_id == reference_ids[-1]
            for index, cache in enumerate(caches):
                assert cache.length == reference_cache.get_seq_length(index) == used_tokens

        # 先判断停止，再准备下一次输入；最终 Token（包括 EOS）不再写入 KV。
        if (not ignore_eos and next_id in eos_ids) or step + 1 == max_new_tokens:
            break
        current_input = torch.tensor([[next_id]], dtype=torch.long, device=input_ids.device)
        if verify:
            reference_input = torch.tensor([[reference_ids[-1]]], dtype=torch.long, device=input_ids.device)

    if verify:
        assert generated_ids == reference_ids
        assert all(cache.length == used_tokens for cache in caches)
        print(f"reference_token_ids = {reference_ids}")
        print("[PASS] 本次逐步 logits、完整 Token 序列与缓存检查通过；不代表性能验证通过")
        if cache_kind == "paged":
            crossed = (used_tokens - 1) // PAGED_BLOCK_SIZE > (tokens - 1) // PAGED_BLOCK_SIZE
            print(f"分页块大小={PAGED_BLOCK_SIZE}，Prefill 长度={tokens}，最终 KV 长度={used_tokens}，Decode 跨块={crossed}")
            if not crossed:
                print("[未覆盖] 本次 Decode 未跨块，需换输入补验；不能据此宣称跨块生成验证通过")
    if timings is not None:
        # 统计在最后Token就绪之后，只查元数据，不计入推理时间；包含已预留的空闲槽位。
        timings["kv_data_bytes"] = sum(t.numel() * t.element_size() for c in caches for t in (c._key, c._value))
        timings["kv_scale_bytes"] = sum(
            t.numel() * t.element_size() for c in caches
            for t in (getattr(c, "_key_scale", None), getattr(c, "_value_scale", None)) if t is not None
        )
        timings["bf16_capacity_bytes"] = sum(c._key.numel() * 4 for c in caches)
        timings["capacity"] = caches[0].capacity
    if cache_kind == "paged":
        for cache in caches:
            cache.release()
        if verify:
            print("已归还全部层分页块；本入口不再重复检查块池内部标记")
    if timings is not None:
        # 终点为最后一个 Token ID 可用；不包含文本解码、打印和函数退出时的缓存释放。
        total = token_ready - started
        timings.update(
            ttft_ms=(first_ready - started) * 1000,
            tpot_ms=(token_ready - first_ready) * 1000 / (len(generated_ids) - 1) if len(generated_ids) > 1 else None,
            total_ms=total * 1000, output_tokens_per_s=len(generated_ids) / total,
        )
    return generated_ids


def benchmark(model, input_ids, *, repeats: int = 3, cache_kind: str = "contiguous", cuda_decode: bool = False, reference_ids: list[int] | None = None, kv_dtype: torch.dtype = torch.bfloat16) -> list[int]:
    """同一模型预热一次、重复纯生成；只打印基础统计，不保存输入或结果文件。"""
    if repeats < 1:
        raise ValueError("测量次数必须为正数")
    backend = "CUDA V0" if cuda_decode else "SDPA"
    print(f"\n基线路径：缓存={cache_kind}，KV={kv_dtype}，Decode={backend}，Prefill=BF16 SDPA")
    print(f"基线：预热=1 次，测量={repeats} 次，输入 Token={input_ids.shape[1]}，输出上限={MAX_NEW_TOKENS}")
    print("范围：模型与输入已就绪，包含 KV 分配和选词；无 HF 对照、分词、文本解码或终端打印")
    print("终点为最后一个 Token ID 可用；不计随后缓存归还。保留实现必需的输入检查和同步，不是内核单独计时。")
    print(f"固定工作量：忽略EOS，生成{MAX_NEW_TOKENS}个Token；EOS后输出不用于质量评估。")
    if cache_kind == "paged":
        print("分页 SDPA 包含逻辑读回与 GQA 临时复制；CUDA Decode 包含块表创建/上传和块号检查同步；两边均包含 KV 追加。")
    # 预热不计入结果；每次调用都新建请求缓存，不复用上个请求的有效内容。
    expected_ids = generate_tokens(model, input_ids, cache_kind=cache_kind, cuda_decode=cuda_decode,
                                   kv_dtype=kv_dtype, ignore_eos=True)
    if reference_ids is not None:
        if kv_dtype == torch.int8:
            print(f"[观测] 与BF16固定长度序列一致={expected_ids == reference_ids}；各自选词，历史可能不同，不是精度验收")
        elif expected_ids != reference_ids:
            raise RuntimeError("CUDA 预热序列与 SDPA 不一致，取消 CUDA 测量；请运行 cuda-check 定位")
        else:
            print("[PASS] 计时外完整 Token 序列与 SDPA 一致；不代表 logits 精度验收通过")
    device = input_ids.device
    torch.cuda.synchronize(device)
    baseline_allocated = torch.cuda.memory_allocated(device)
    rows, after_allocated = [], []
    print(f"预热后活跃张量显存={baseline_allocated / 1024**2:.2f} MiB")
    for run in range(repeats):
        # 重置峰值与读取显存放在生成计时区间外；不调用 empty_cache 改变分配器状态。
        torch.cuda.reset_peak_memory_stats(device)
        metrics = {}
        generated_ids = generate_tokens(model, input_ids, timings=metrics, cache_kind=cache_kind,
                                        cuda_decode=cuda_decode, kv_dtype=kv_dtype, ignore_eos=True)
        torch.cuda.synchronize(device)
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        peak = torch.cuda.max_memory_allocated(device)
        # 结果一致性检查也在计时之外；这只验证重复运行，不代替 HF 正确性对照。
        if generated_ids != expected_ids:
            raise RuntimeError("重复请求 Token 序列不一致，停止性能汇总")
        rows.append(metrics)
        after_allocated.append(allocated)
        if run == 0:
            kv_bytes = metrics["kv_data_bytes"] + metrics["kv_scale_bytes"]
            ratio = metrics["bf16_capacity_bytes"] / kv_bytes
            print(f'[容量] 每层物理容量={metrics["capacity"]} Token，全部层KV数据={metrics["kv_data_bytes"]}字节，scale={metrics["kv_scale_bytes"]}字节，总计={kv_bytes}字节')
            print(f'[容量] 同容量BF16字节数/当前KV字节数={ratio:.6f}；不含权重、页表、临时张量和分配器保留空间，不代表最大上下文倍数')
        tpot = "N/A" if metrics["tpot_ms"] is None else f'{metrics["tpot_ms"]:.3f} ms'
        print(f'测量 {run + 1}：输出={len(generated_ids)}，TTFT={metrics["ttft_ms"]:.3f} ms，平均 TPOT={tpot}，总耗时={metrics["total_ms"]:.3f} ms，输出吞吐={metrics["output_tokens_per_s"]:.3f} Token/s')
        print(f"显存：结束 allocated={allocated / 1024**2:.2f} MiB（较预热 {allocated - baseline_allocated:+d} 字节），reserved={reserved / 1024**2:.2f} MiB，峰值 allocated={peak / 1024**2:.2f} MiB")
    for name in ("ttft_ms", "tpot_ms", "total_ms", "output_tokens_per_s"):
        values = [row[name] for row in rows if row[name] is not None]
        print(f'{name} 中位数 = {statistics.median(values):.3f}' if values else f"{name} 中位数 = N/A（仅生成一个 Token）")
    if all(value == baseline_allocated for value in after_allocated):
        print("[观测] 本次各请求结束后的活跃张量显存均回到预热基线；不代表长期无泄漏")
    else:
        print("[注意] 请求结束后的活跃张量显存未全部回到预热基线，请结合逐次变化检查；不能仅凭此认定泄漏")
    print("reserved 是分配器保留空间，峰值包含模型权重；这些数值不是整卡总显存。少量重复的中位数不是 P95 或服务吞吐。")
    return generated_ids


@torch.inference_mode()
def check_cuda_decode(model, input_ids) -> None:
    """两套独立分页缓存做有界贪心生成；首次选词不一致即停止，不强制喂参考 Token。"""
    config = model.config
    if config.head_dim != 128 or PAGED_BLOCK_SIZE != 16:
        raise ValueError("CUDA V0 只支持每头 128 维和块大小 16")
    tokens = input_ids.shape[1]
    num_blocks = (tokens + MAX_NEW_TOKENS + PAGED_BLOCK_SIZE - 1) // PAGED_BLOCK_SIZE
    eos_ids = model.generation_config.eos_token_id
    eos_ids = [eos_ids] if isinstance(eos_ids, int) else (eos_ids or [])
    actual_ids, reference_ids = [], []
    # 两条路径共享只读权重，但不共享缓存张量或请求元数据。
    actual_caches, reference_caches = [], []
    try:
        for caches in (actual_caches, reference_caches):
            for _ in model.model.layers:
                caches.append(PagedKVCache(PagedKVStorage(
                    config.num_key_value_heads, config.head_dim, num_blocks,
                    PAGED_BLOCK_SIZE, device=input_ids.device,
                )))
        all_caches = actual_caches + reference_caches
        actual_input, reference_input = input_ids, input_ids
        print(f"对照上限={MAX_NEW_TOKENS} 个新 Token，输入长度={tokens}，每层物理容量={num_blocks * PAGED_BLOCK_SIZE}")
        for step in range(MAX_NEW_TOKENS):
            is_prefill = step == 0
            # 第一次 Decode 的位置为 tokens；此后每步只增加一个位置。
            positions = torch.arange(tokens, device=input_ids.device).unsqueeze(0) if is_prefill else torch.tensor([[tokens + step - 1]], device=input_ids.device)
            # 各自完整走一遍模型，上一层自己的输出直接进入下一层，不能换回参考状态。
            actual = model_forward(model, actual_input, positions, actual_caches, is_prefill=is_prefill, cuda_decode=not is_prefill)
            reference = model_forward(model, reference_input, positions, reference_caches, is_prefill=is_prefill)
            assert torch.isfinite(actual).all().item() and torch.isfinite(reference).all().item()
            used = tokens + step
            assert all(cache.length == used for cache in all_caches)
            actual_id, reference_id = actual.argmax(dim=-1).item(), reference.argmax(dim=-1).item()
            if is_prefill:
                # 起点必须相同，防止把 Prefill 的差异误归因于 CUDA Decode。
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                print(f"[PASS] 两套 SDPA Prefill logits 完全一致：首个 Token={actual_id}，KV 长度={used}")
            else:
                # BF16 logits 的差值转 FP32 统计；不直接套用 Attention 算子的容差。
                difference = actual.float() - reference.float()
                reference_norm = torch.linalg.vector_norm(reference.float()).item()
                relative_l2 = f"{torch.linalg.vector_norm(difference).item() / reference_norm:.8g}" if reference_norm != 0 else "N/A（参考全零）"
                print(f"[观测] Decode 第 {step} 步 logits：最大绝对误差={difference.abs().max().item():.8g}，相对 L2 误差={relative_l2}")
            actual_ids.append(actual_id)
            reference_ids.append(reference_id)
            if actual_id != reference_id:
                print(f"[分歧] 第 {step + 1} 个 Token：CUDA={actual_id}，SDPA={reference_id}，KV 长度={used}")
                # 只在分歧时补充候选，避免每步都打印大量词表诊断。
                # 这是原始 logits 的间隔，不是概率；topk 的同分排序不保证与 argmax 相同。
                for label, logits in (("CUDA", actual), ("SDPA", reference)):
                    scores, ids = logits[0, 0].float().topk(2)
                    print(f"[诊断] {label} 前两名 ID={ids.tolist()}，logits={scores.tolist()}，间隔={(scores[0] - scores[1]).item():.8g}")
                print(f"CUDA 序列={actual_ids}，SDPA 序列={reference_ids}")
                raise AssertionError("首次贪心选词不一致，停止后续生成；未将参考 Token 灌入 CUDA 路径")
            # 最终选出的 Token（包括 EOS）不再写入缓存，最终长度为 P + N - 1。
            if actual_id in eos_ids or step + 1 == MAX_NEW_TOKENS:
                break
            actual_input = torch.tensor([[actual_id]], dtype=torch.long, device=input_ids.device)
            reference_input = torch.tensor([[reference_id]], dtype=torch.long, device=input_ids.device)
        assert actual_ids == reference_ids
        crossed = (used - 1) // PAGED_BLOCK_SIZE > (tokens - 1) // PAGED_BLOCK_SIZE
        reason = "EOS" if actual_ids[-1] in eos_ids else "达到新 Token 上限"
        print(f"CUDA 序列={actual_ids}")
        print(f"SDPA 序列={reference_ids}")
        print(f"停止原因={reason}，生成数={len(actual_ids)}，Decode 次数={len(actual_ids) - 1}，最终 KV 长度={used}，Decode 跨块={crossed}")
        if len(actual_ids) == 1:
            print("[未覆盖] 首个 Token 为 EOS，本次未执行 CUDA Decode")
        elif not crossed:
            print("[未覆盖] 本次真实模型 Decode 没有跨块")
    finally:
        # 成功、断言失败或首个 Token 为 EOS，都归还已经建立的两套缓存。
        for cache in actual_caches + reference_caches:
            cache.release()
        print("已归还本次两套分页缓存；本入口不再重复检查块池内部标记")
    print("[PASS] 本次有界生成的 Token 序列与自建 SDPA 参考一致，缓存检查通过")
    print("[观测] 未设完整 logits 精度阈值；本次结果不代表独立 HF 对照、多输入、长上下文或性能验证通过")


@torch.inference_mode()
def check_int8_decode(model, input_ids) -> None:
    """共享BF16 Prefill，按BF16选词驱动两条CUDA路径；仅同历史对照，不测性能。"""
    config = model.config
    length = input_ids.shape[1]
    blocks = (length + MAX_NEW_TOKENS - 1 + PAGED_BLOCK_SIZE - 1) // PAGED_BLOCK_SIZE
    reference_caches, quantized_caches = [], []
    positions = torch.arange(length, device=input_ids.device)[None]
    hidden = model.model.embed_tokens.weight[input_ids]
    try:
        # 只走一次BF16 Prefill，逐层将同一份RoPE后的K和原始V保存成两种表示。
        # INT8缓存不参与Prefill Attention，避免起点已经是两套不同的隐藏状态。
        for layer in model.model.layers:
            for caches, dtype in ((reference_caches, torch.bfloat16), (quantized_caches, torch.int8)):
                storage = PagedKVStorage(config.num_key_value_heads, config.head_dim, blocks,
                                        PAGED_BLOCK_SIZE, device=input_ids.device, kv_dtype=dtype)
                caches.append(PagedKVCache(storage))
            hidden = decoder_layer_forward(hidden, layer, positions, reference_caches[-1], config, is_prefill=True)
            # get只在诊断准备阶段复制当前层历史；Decode直接由CUDA读取物理块。
            key, value = reference_caches[-1].get()
            quantized_caches[-1].append(key, value, fused=True)
        first_logits = final_logits(model, hidden)
        # 对照入口保留数值底线，避免NaN经argmax后被误报为选词一致。
        assert torch.isfinite(first_logits).all().item()
        first_token = first_logits.argmax(dim=-1)
        all_caches = reference_caches + quantized_caches
        assert all(cache.length == length for cache in all_caches)
        print(f"[PASS] 共用BF16 Prefill完成：层数={len(reference_caches)}，KV长度={length}，首Token={first_token.item()}")
        # 首Token来自共同Prefill，不计入Decode选词一致率；若它是EOS则零步结束。
        del hidden, first_logits, key, value
        eos_ids = model.generation_config.eos_token_id
        eos_ids = [eos_ids] if isinstance(eos_ids, int) else (eos_ids or [])
        current_id = first_token.item()
        compared = matched = 0
        relative_errors = []
        kl_values, mismatch_gaps = [], []
        first_mismatch = None
        used = length
        for step in range(MAX_NEW_TOKENS - 1):
            if current_id in eos_ids:
                break
            # 两套缓存只共享Token历史，不共享中间隐藏状态或新计算的K/V。
            current_input = input_ids.new_tensor([[current_id]])
            positions = input_ids.new_tensor([[length + step]])
            reference = model_forward(model, current_input, positions, reference_caches,
                                      is_prefill=False, cuda_decode=True)
            actual = model_forward(model, current_input, positions, quantized_caches,
                                   is_prefill=False, cuda_decode=True)
            assert torch.isfinite(reference).all().item() and torch.isfinite(actual).all().item()
            reference_scores, actual_scores = reference.float().flatten(), actual.float().flatten()
            delta = actual_scores - reference_scores
            relative_errors.append((delta.norm() / reference_scores.norm().clamp_min(1e-12)).item())
            # 完整词表、温度1、自然对数：KL方向为BF16到INT8，不做Top-k截断。
            reference_logp = reference_scores.log_softmax(dim=-1)
            actual_logp = actual_scores.log_softmax(dim=-1)
            kl = (reference_logp.exp() * (reference_logp - actual_logp)).sum().item()
            kl_values.append(kl)  # 保留FP32原始结果；极小负值可能来自浮点舍入。
            reference_id = reference.argmax(dim=-1).item()
            actual_id = actual.argmax(dim=-1).item()
            compared += 1
            matched += int(reference_id == actual_id)
            if reference_id != actual_id:
                # 衡量实际竞争候选的差距，而不假设INT8选择的是BF16第二名。
                gap = (reference_scores[reference_id] - reference_scores[actual_id]).item()
                mismatch_gaps.append(gap)
                if first_mismatch is None:
                    top_two = reference_scores.topk(2).values
                    margin = (top_two[0] - top_two[1]).item()
                    # 并列使用竞争排名：严格更高的候选数+1，不对整个词表排序。
                    rank = (reference_scores > reference_scores[actual_id]).sum().item() + 1
                    ids = [reference_id, actual_id]
                    # 只保存CPU标量与两个候选分数，不保留逐步GPU logits。
                    first_mismatch = (step + 2, length + step, ids, kl, margin, rank,
                                      reference_scores[ids].tolist(), actual_scores[ids].tolist())
            # INT8即使选到EOS也不提前结束；下一轮始终服从BF16基线，避免文本历史分叉。
            current_id = reference_id
            used = length + compared
        reason = "BF16 EOS" if current_id in eos_ids else "达到输出上限"
        print(f"[汇总] 同历史对照：输出上限={MAX_NEW_TOKENS}，基线输出数={compared + 1}，Decode比较数={compared}，停止原因={reason}")
        if compared:
            print(f"[观测] Decode选词一致={matched}/{compared}（{matched / compared:.2%}），不含共用首Token")
            print(f"[观测] logits相对L2：均值={statistics.mean(relative_errors):.8g}，最大值={max(relative_errors):.8g}")
            print(f"[观测] KL(BF16 || INT8)：均值={statistics.mean(kl_values):.8g}，最大值={max(kl_values):.8g}（自然对数，温度1）")
            if mismatch_gaps:
                print(f"[观测] 分歧步BF16竞争分差：均值={statistics.mean(mismatch_gaps):.8g}，最大值={max(mismatch_gaps):.8g}，仅统计{len(mismatch_gaps)}个分歧步")
        else:
            print("[未覆盖] 首Token为EOS，没有执行Decode；一致率与误差无定义")
        if first_mismatch is not None:
            number, position, ids, kl, margin, rank, reference_pair, actual_pair = first_mismatch
            print(f"[观测] 首次分歧：第{number}个输出Token，输入位置={position}，BF16={ids[0]}，INT8={ids[1]}")
            print(f"[观测] 该步KL={kl:.8g}，BF16前两名分差={margin:.8g}，INT8所选Token在BF16中排名={rank}（严格更高数+1）")
            print(f"[观测] 候选顺序={ids}，BF16分数={reference_pair}，INT8分数={actual_pair}")
        elif compared:
            print("[观测] 本次Decode比较未出现选词分歧")
        crossed = compared > 0 and (used - 1) // PAGED_BLOCK_SIZE > (length - 1) // PAGED_BLOCK_SIZE
        assert all(cache.length == used for cache in all_caches)
        print(f"[PASS] 两套全部层KV长度={used}，Decode跨块={crossed}")
    finally:
        for cache in reference_caches + quantized_caches:
            cache.release()
    assert all(cache._pool.num_free_blocks == blocks for cache in reference_caches + quantized_caches)
    print("[PASS] 两套缓存块全部归还；本轮为BF16驱动的同历史对照，不代表INT8独立生成、完整精度验收、HF对照或性能通过")


def main() -> None:
    """加载与分词只做一次；选择对照或纯生成模式，最后统一展示结果。"""
    parser = argparse.ArgumentParser(description="Qwen3 自建 BF16 生成与正确性对照")
    # 默认保留对照行为；纯生成须显式选择，避免把输出文本误认为验证通过。
    parser.add_argument("--mode", choices=("check", "generate", "benchmark", "cuda-check", "int8-check"), default="check", help="check：HF 对照；generate：纯生成；benchmark：纯生成基线；cuda-check：有界 CUDA Decode 对照；int8-check：BF16驱动的INT8同历史对照")
    parser.add_argument("--repeats", type=int, default=3, help="benchmark 模式测量次数，默认 3，另有 1 次预热")
    parser.add_argument("--cache", choices=("contiguous", "paged"), default="contiguous", help="默认连续缓存；paged 为每块 16 Token 的分页参考")
    parser.add_argument("--kv-dtype", choices=("bf16", "int8"), default="bf16",
                        help="生成的KV类型；benchmark选int8则比较BF16/INT8 CUDA，均需paged；诊断模式自行选择")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats 必须为正数")
    if args.mode in ("cuda-check", "int8-check") and args.cache != "paged":
        parser.error("CUDA 对照模式需要显式指定 --cache paged")
    if args.kv_dtype == "int8" and (args.mode not in ("generate", "benchmark") or args.cache != "paged"):
        parser.error("--kv-dtype int8 仅用于 generate/benchmark 且需 --cache paged")
    if not torch.cuda.is_available():
        raise RuntimeError("模型生成需要在云端 CUDA 环境运行")
    text = input("请输入一段文本：")
    if not text.strip():
        raise ValueError("输入不能为空")
    # 输入只保留在进程内；不保存文本，也不增加下载或配置文件。
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    formatted_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    input_ids = tokenizer(formatted_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to("cuda")
    print(f"正在从本地缓存加载固定版本 Qwen3-8B，模式={args.mode}，缓存={args.cache}……")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True,
    ).to("cuda")
    model.eval()
    # 固定模型结构仅在加载后检查一次，不在36层的每次前向里重复判断。
    if model.config.hidden_act != "silu" or any(layer.self_attn.sliding_window is not None for layer in model.model.layers):
        raise ValueError("当前只支持SiLU且无滑动窗口的Qwen3")
    if args.mode == "int8-check":
        check_int8_decode(model, input_ids)
        return  # 同历史对照不等于INT8独立生成，不进入普通生成输出。
    if args.mode == "cuda-check":
        check_cuda_decode(model, input_ids)
        return  # 对照入口自行汇总，不能落入下面的普通生成结果打印。
    if args.mode == "benchmark":
        if args.kv_dtype == "int8":
            # 同一模型、同一输入、相同输出步数；先BF16再INT8，固定顺序仍可能受时钟漂移影响。
            # 这里不执行SDPA陪跑；质量检查由int8-check承担，量化写入开销包含在计时内。
            print("配对基线：BF16 CUDA → INT8 CUDA；各自独立选词，不要求跨精度序列完全一致。")
            reference_ids = benchmark(model, input_ids, repeats=args.repeats, cache_kind="paged", cuda_decode=True)
            benchmark(model, input_ids, repeats=args.repeats, cache_kind="paged", cuda_decode=True,
                      kv_dtype=torch.int8, reference_ids=reference_ids)
            print("[完成] 固定长度BF16/INT8 CUDA初步基线；不代表正式精度或稳定加速比验收")
            return
        generated_ids = benchmark(model, input_ids, repeats=args.repeats, cache_kind=args.cache)
        if args.cache == "paged":
            # 一次加载、同一输入、同一上限；两条路径仍分别创建并归还自己的缓存。
            # 先 SDPA 后 CUDA 仅用于初步基线，固定顺序和少量重复不能代表稳定加速比。
            generated_ids = benchmark(
                model, input_ids, repeats=args.repeats, cache_kind="paged",
                cuda_decode=True, reference_ids=generated_ids,
            )
        return  # benchmark忽略EOS，不把固定工作量输出打印为正常回答或EOS停止。
    else:
        # INT8直接复用循环，由自身logits选词；不创建BF16陪跑缓存。
        use_int8 = args.kv_dtype == "int8"
        if args.mode == "generate":
            print(f"生成路径：KV={args.kv_dtype}，Prefill=BF16 SDPA，Decode={'INT8 CUDA' if use_int8 else 'BF16 SDPA'}")
        generated_ids = generate_tokens(
            model, input_ids, verify=(args.mode == "check"), cache_kind=args.cache,
            cuda_decode=use_int8, kv_dtype=torch.int8 if use_int8 else torch.bfloat16,
        )
    eos_ids = model.generation_config.eos_token_id
    eos_ids = [eos_ids] if isinstance(eos_ids, int) else (eos_ids or [])
    reason = "EOS" if generated_ids[-1] in eos_ids else "达到新 Token 上限"
    print(f"generated_token_ids = {generated_ids}")
    print(f"generated_text = {tokenizer.decode(generated_ids, skip_special_tokens=True)!r}")
    print(f"停止原因={reason}，生成数={len(generated_ids)}，Decode 次数={len(generated_ids) - 1}")
    if len(generated_ids) == 1:
        print("首个 Token 为 EOS，本次未覆盖 Decode")
    if args.mode == "generate":
        print("纯生成完成；未执行 HF 对照或性能测量")


if __name__ == "__main__":
    main()
