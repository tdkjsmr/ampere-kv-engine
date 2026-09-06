"""使用 Hugging Face Qwen3-8B 演示 Prefill 与复用 KV Cache 的逐 Token 生成。"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


# 模型与分词器使用同一个固定版本，避免远端更新改变参考结果。
MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
# 先用很短的输出观察生成过程；达到上限时，文本可能尚未形成完整句子。
MAX_NEW_TOKENS = 8


def main() -> None:
    """手写循环生成最多 8 个 Token，并与 Hugging Face 的生成结果对照。"""

    if not torch.cuda.is_available():
        raise RuntimeError("此参考程序需要在云端 CUDA 环境运行")

    # 输入只存在于当前进程，不把实际文本写入源码或结果文件。
    text = input("请输入一段文本：")
    if not text.strip():
        raise ValueError("输入不能为空")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)

    # 使用模型自带的对话模板，补上回答起始标记，并关闭 thinking 模式。
    formatted_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    # 模板已包含特殊标记，分词时不再重复添加；同时取得 attention_mask。
    inputs = tokenizer(formatted_text, add_special_tokens=False, return_tensors="pt")

    print("正在加载固定版本 Qwen3-8B，首次运行需要下载模型权重……")
    # BF16 保存权重；使用 PyTorch 自带的 SDPA，不额外引入 Attention 库。
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
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
        for step in range(MAX_NEW_TOKENS):
            # 最后一个位置预测下一个 Token；保留 [批大小=1, 输入长度=1] 的形状。
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            next_token_id = next_token.item()
            generated_ids.append(next_token_id)

            # 先检查终止条件，避免为 EOS 或第 8 个 Token 再执行无用的前向计算。
            if next_token_id in eos_ids or step + 1 == MAX_NEW_TOKENS:
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
