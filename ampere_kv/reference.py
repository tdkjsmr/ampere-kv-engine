"""使用 Hugging Face Qwen3-8B 演示 Prefill 与复用 KV Cache 的逐 Token 生成。"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


# 模型与分词器使用同一个固定版本，避免远端更新改变参考结果。
MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
# 先用很短的输出观察生成过程；达到上限时，文本可能尚未形成完整句子。
MAX_NEW_TOKENS = 8


def inspect_kv_cache(cache, config, expected_tokens: int, stage: str) -> None:
    """观察当前单请求的 BF16 动态缓存，只读取形状和数据量，不修改缓存。"""

    # 按模型配置核对每一层；缓存存的是 KV 头，不是扩展后的 Query 头。
    expected_shape = (1, config.num_key_value_heads, expected_tokens, config.head_dim)
    if len(cache) != config.num_hidden_layers:
        raise RuntimeError(f"[FAIL] {stage}：KV Cache 层数与模型配置不一致")
    actual_bytes = 0
    for layer_index, (key, value) in enumerate(cache):
        # 四个维度依次是：批大小、KV 头数、缓存长度、每头维度。
        if tuple(key.shape) != expected_shape or tuple(value.shape) != expected_shape:
            raise RuntimeError(f"[FAIL] {stage}：第 {layer_index} 层 K/V 形状不符合 {expected_shape}")
        if key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16:
            raise RuntimeError(f"[FAIL] {stage}：第 {layer_index} 层 K/V 不是 BF16")
        # numel 是元素数量，element_size 是每个元素的字节数；K 和 V 都要计入。
        actual_bytes += key.numel() * key.element_size() + value.numel() * value.element_size()
        if layer_index == 0:
            print(f"{stage} 第一层：K={tuple(key.shape)}，V={tuple(value.shape)}，类型={key.dtype}")

    # 两个 2 分别表示 K/V 两份数据、BF16 每个元素占 2 字节。
    bytes_per_token = config.num_hidden_layers * 2 * config.num_key_value_heads * config.head_dim * 2
    expected_bytes = bytes_per_token * expected_tokens
    print(f"{stage}：每 Token 理论 KV 数据量={bytes_per_token} 字节")
    print(f"{stage}：全部层 KV 实际数据量={actual_bytes} 字节，理论值={expected_bytes} 字节")
    # 这里只统计张量数据，不包含模型权重、临时张量或显存分配器的保留空间。
    if actual_bytes != expected_bytes:
        raise RuntimeError(f"[FAIL] {stage}：KV 实际数据量与理论值不一致")
    print(f"[PASS] {stage}：全部层 KV 形状、类型和数据量核对通过")


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
        # Prefill：计算完整输入的 K/V，并给出第一个输出 Token 的分数。
        outputs = model(**inputs, use_cache=True)
        # 保存整数长度，不保留额外的缓存引用；避免影响后面对照生成时释放缓存。
        prompt_tokens = inputs["input_ids"].shape[1]
        inspect_kv_cache(outputs.past_key_values, model.config, prompt_tokens, "Prefill")
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
            # Decode：复用已经计算的 K/V，只为刚生成的一个 Token 计算新的 K/V。
            # 当前单条无填充输入的位置由模型根据缓存长度自动推导。
            outputs = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=outputs.past_key_values,
                use_cache=True,
            )
            # 仅观察第一次 Decode：刚选出的 Token 经前向计算后才真正加入缓存。
            # 检查所有层的长度均为输入长度 + 1，而不只检查第一层的显示结果。
            if step == 0:
                inspect_kv_cache(outputs.past_key_values, model.config, prompt_tokens + 1, "首次 Decode")

        # 删除持有旧 KV Cache 的结果对象，让显存可供下一次生成复用。
        # 模型权重继续复用；对照从原始输入重新开始，不传入旧缓存。
        del outputs
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
    print("[PASS] 手写生成循环与 Hugging Face 的 Token ID 完全一致")


if __name__ == "__main__":
    main()
