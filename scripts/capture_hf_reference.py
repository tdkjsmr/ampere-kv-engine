"""生成固定 Prompt 的 Hugging Face Greedy 与 Logits 基线。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ampere_kv.config import load_yaml_mapping, require_mapping


def _sha256(path: Path) -> str:
    """计算输出 Tensor 文件的 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    """在锁定模型版本上生成八个可重复的 Reference。"""

    parser = argparse.ArgumentParser(description="生成 Qwen3-8B HF Reference")
    parser.add_argument("--lock", required=True, help="模型锁文件")
    parser.add_argument("--prompts", required=True, help="固定 Prompt JSON")
    parser.add_argument("--output", required=True, help="Reference Manifest JSON")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("HF Reference 必须在云端 CUDA 环境生成")

    lock = load_yaml_mapping(args.lock)
    model_lock = require_mapping(lock, "model")
    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    if not isinstance(prompts, list) or len(prompts) != 8:
        raise ValueError("固定 Prompt 文件必须恰好包含 8 条记录")

    tokenizer = AutoTokenizer.from_pretrained(
        model_lock["id"],
        revision=model_lock["tokenizer_revision"],
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_lock["id"],
        revision=model_lock["revision"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    ).to("cuda")
    model.eval()

    output_path = Path(args.output)
    tensor_dir = output_path.parent / f"{output_path.stem}_tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)

    records = []
    with torch.inference_mode():
        for prompt in prompts:
            prompt_id = prompt["id"]
            inputs = tokenizer.apply_chat_template(
                prompt["messages"],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
            ).to("cuda")

            forward = model(**inputs, use_cache=True)
            last_logits = forward.logits[:, -1, :].float().cpu()
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )
            input_length = inputs["input_ids"].shape[1]
            generated_ids = generated[:, input_length:].cpu()
            generated_text = tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True,
            )

            tensor_path = tensor_dir / f"{prompt_id}.pt"
            torch.save(
                {
                    "last_logits": last_logits,
                    "generated_ids": generated_ids,
                },
                tensor_path,
            )
            records.append(
                {
                    "id": prompt_id,
                    "input_length": input_length,
                    "generated_ids": generated_ids[0].tolist(),
                    "generated_text": generated_text,
                    "tensor_file": str(tensor_path.relative_to(output_path.parent)),
                    "tensor_sha256": _sha256(tensor_path),
                }
            )

    torch.cuda.synchronize()
    manifest = {
        "schema_version": 1,
        "model_id": model_lock["id"],
        "model_revision": model_lock["revision"],
        "tokenizer_revision": model_lock["tokenizer_revision"],
        "dtype": "bfloat16",
        "sampling": {
            "strategy": "greedy",
            "do_sample": False,
            "enable_thinking": False,
            "max_new_tokens": args.max_new_tokens,
        },
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[PASS] HF Reference 已写入：{output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
