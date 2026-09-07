"""使用 Qwen3-8B 对照静态与动态 KV Cache，观察预分配容量和有效长度。"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, StaticCache


# 模型与分词器使用同一个固定版本，避免远端更新改变参考结果。
MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
# 先用很短的输出观察生成过程；达到上限时，文本可能尚未形成完整句子。
MAX_NEW_TOKENS = 8


def inspect_kv_cache(cache: StaticCache, config, expected_tokens: int, stage: str) -> None:
    """核对单请求 BF16 静态缓存，区分预分配容量和已经写入的有效长度。"""

    # 静态张量的第三维是容量，不会随着 Decode 增长；KV 头也不是 Query 头。
    capacity = cache.get_max_cache_shape()
    if not 0 <= expected_tokens <= capacity:
        raise RuntimeError(f"[FAIL] {stage}：有效长度超出缓存容量")
    expected_shape = (1, config.num_key_value_heads, capacity, config.head_dim)
    if len(cache.key_cache) != config.num_hidden_layers or len(cache.value_cache) != config.num_hidden_layers:
        raise RuntimeError(f"[FAIL] {stage}：KV Cache 层数与模型配置不一致")
    actual_bytes = 0
    for layer_index, (key, value) in enumerate(zip(cache.key_cache, cache.value_cache)):
        # 四个维度依次是：批大小、KV 头数、预分配容量、每头维度。
        if tuple(key.shape) != expected_shape or tuple(value.shape) != expected_shape:
            raise RuntimeError(f"[FAIL] {stage}：第 {layer_index} 层 K/V 形状不符合 {expected_shape}")
        if key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16:
            raise RuntimeError(f"[FAIL] {stage}：第 {layer_index} 层 K/V 不是 BF16")
        # HF 4.51 的接口通过第一头的非零 K 位置估算已写入长度，仅用于本次诊断。
        # 它不是任意数据都适用的长度计数器；真正的写入位置由下面的循环显式管理。
        if int(cache.get_seq_length(layer_index).item()) != expected_tokens:
            raise RuntimeError(f"[FAIL] {stage}：第 {layer_index} 层缓存观测长度不符合 {expected_tokens}")
        # numel 是元素数量，element_size 是每个元素的字节数；K 和 V 都要计入。
        actual_bytes += key.numel() * key.element_size() + value.numel() * value.element_size()
        if layer_index == 0:
            print(f"{stage} 第一层：K={tuple(key.shape)}，V={tuple(value.shape)}，类型={key.dtype}")

    # 两个 2 分别表示 K/V 两份数据、BF16 每个元素占 2 字节。
    bytes_per_token = config.num_hidden_layers * 2 * config.num_key_value_heads * config.head_dim * 2
    expected_bytes = bytes_per_token * capacity
    print(f"{stage}：容量={capacity} Token，有效长度={expected_tokens} Token")
    print(f"{stage}：每 Token 理论 KV 数据量={bytes_per_token} 字节")
    print(f"{stage}：全部层 KV 已分配数据量={actual_bytes} 字节，理论值={expected_bytes} 字节")
    print(f"{stage}：按有效长度计算的 KV 数据量={bytes_per_token * expected_tokens} 字节")
    # 这里只统计张量数据，不包含模型权重、临时张量或显存分配器的保留空间。
    if actual_bytes != expected_bytes:
        raise RuntimeError(f"[FAIL] {stage}：KV 实际数据量与理论值不一致")
    print(f"[PASS] {stage}：全部层 KV 容量、观测长度、类型和数据量核对通过")


def main() -> None:
    """手写循环生成最多 8 个 Token，并与 Hugging Face 的生成结果对照。"""

    if not torch.cuda.is_available():
        raise RuntimeError("此参考程序需要在云端 CUDA 环境运行")

    # 输入只存在于当前进程，不把实际文本写入源码或结果文件。
    text = input("请输入一段文本：")
    if not text.strip():
        raise ValueError("输入不能为空")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)

    # 使用模型自带的对话模板，补上回答起始标记，并关闭 thinking 模式。
    formatted_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    # 模板已包含特殊标记，分词时不再重复添加；同时取得 attention_mask。
    inputs = tokenizer(formatted_text, add_special_tokens=False, return_tensors="pt")

    print("正在从本地缓存加载固定版本 Qwen3-8B……")
    # BF16 保存权重；使用 PyTorch 自带的 SDPA，不额外引入 Attention 库。
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to("cuda")
    model.eval()
    inputs = inputs.to("cuda")

    # 模型可能定义多个结束标记，统一成列表后检查，避免漏掉其中一种。
    eos_ids = model.generation_config.eos_token_id
    eos_ids = [eos_ids] if isinstance(eos_ids, int) else (eos_ids or [])
    generated_ids = []
    attention_mask = inputs["attention_mask"]

    # 推理不记录梯度。第一次输入完整文本，后续每次只输入刚生成的 Token。
    with torch.inference_mode():
        prompt_tokens = inputs["input_ids"].shape[1]
        # 只为本次输入和最多 8 个输出预留空间，不按模型最大上下文分配。
        # 最后选出的 Token 不再送回模型，所以正常生成 8 个 Token 会剩一个空位。
        cache = StaticCache(
            config=model.config,
            max_batch_size=1,
            max_cache_len=prompt_tokens + MAX_NEW_TOKENS,
            device=inputs["input_ids"].device,
            dtype=torch.bfloat16,
        )
        # Prefill 写入位置 0 到输入长度减 1；HF 负责屏蔽容量中尚未写入的位置。
        cache_position = torch.arange(prompt_tokens, device=inputs["input_ids"].device)
        # Prefill：计算完整输入的 K/V，并给出第一个输出 Token 的分数。
        outputs = model(**inputs, past_key_values=cache, cache_position=cache_position, use_cache=True)
        used_tokens = prompt_tokens
        inspect_kv_cache(cache, model.config, used_tokens, "Prefill")
        for step in range(MAX_NEW_TOKENS):
            # 最后一个位置预测下一个 Token；保留 [批大小=1, 输入长度=1] 的形状。
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            next_token_id = next_token.item()
            generated_ids.append(next_token_id)

            # 先检查终止条件，避免为 EOS 或第 8 个 Token 再执行无用的前向计算。
            if next_token_id in eos_ids or step + 1 == MAX_NEW_TOKENS:
                if step == 0:
                    print("未执行 Decode：首个 Token 已满足终止条件，跳过缓存增长检查")
                break

            # 掩码覆盖历史缓存和本次新输入，所以每次 Decode 都要在末尾补一个 1。
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(attention_mask[:, :1])], dim=1
            )
            # 下一个空位置恰好等于有效长度；先检查边界，再写入单个 Token 的 K/V。
            if used_tokens >= cache.max_cache_len:
                raise RuntimeError("[FAIL] Decode 写入位置超出静态缓存容量")
            cache_position = torch.tensor([used_tokens], device=inputs["input_ids"].device)
            # Decode 复用同一块存储；单条无填充输入的 RoPE 位置也由此位置推导。
            outputs = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=cache,
                cache_position=cache_position,
                use_cache=True,
            )
            used_tokens += 1
            # 仅观察第一次 Decode：刚选出的 Token 经前向计算后才真正加入缓存。
            # 检查所有层的长度均为输入长度 + 1，而不只检查第一层的显示结果。
            if step == 0:
                inspect_kv_cache(cache, model.config, used_tokens, "首次 Decode")

        print(f"生成结束：静态缓存有效长度={used_tokens}，容量={cache.max_cache_len}")
        # 两个对象都引用了静态缓存，必须一起释放；保留权重供动态缓存对照使用。
        del outputs, cache
        # 新建内存中的配置，避免继承模型默认的采样、重复惩罚等设置。
        greedy_config = GenerationConfig(
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.0,
            eos_token_id=eos_ids or None,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
        # 不传入静态缓存；Qwen3 默认新建 DynamicCache，从相同原始输入开始对照。
        reference = model.generate(
            **inputs,
            generation_config=greedy_config,
            # 禁止模型默认的采样设置覆盖上面指定的贪心配置。
            use_model_defaults=False,
            # 与手写前向一致，保留全部位置的 logits，避免改变输出投影的计算形状。
            logits_to_keep=0,
        )
        # generate 返回“原始输入 + 新 Token”；只截取新增部分参与比较，保留 EOS。
        reference_ids = reference[0, inputs["input_ids"].shape[1]:].tolist()

    # 一起解码生成的 Token，避免单个 Token 恰好只有部分汉字字节时显示不完整。
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"generated_token_ids = {generated_ids}")
    print(f"generated_text = {generated_text!r}")
    print(f"reference_token_ids = {reference_ids}")
    # 比较完整 ID 列表，同时检查 Token 内容、生成长度和结束位置。
    if generated_ids != reference_ids:
        raise RuntimeError("[FAIL] 手写生成循环与 Hugging Face 的 Token ID 不一致")
    print("[PASS] 静态缓存手写循环与 HF 动态缓存的 Token ID 完全一致")


if __name__ == "__main__":
    main()
