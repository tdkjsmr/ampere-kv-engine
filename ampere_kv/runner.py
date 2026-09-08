"""自建 Qwen3 执行路径的起点：真实 Embedding 与第一层 RMSNorm 对照，暂不生成文本。"""

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
    # 会加载完整 BF16 模型，但只执行 Embedding 和第一层入口归一化，不跑完整前向。
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


if __name__ == "__main__":
    main()
