"""固定 Token ID 的离线整批基线；每个进程只加载一种引擎。"""

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path


MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
VLLM_VERSION = "0.29.0"
BLOCK_SIZE = 16


def load_inputs(path: Path, batch: int, input_tokens: int, smoke: bool):
    """输入文件只在内存中展开；报告只保存文件和实际选中 ID 的摘要。"""
    raw = path.read_bytes()
    prompts = json.loads(raw)["prompts"]
    if len(prompts) < batch:
        raise ValueError(f"需要 {batch} 条输入，文件只有 {len(prompts)} 条")
    selected = prompts[:batch]
    if smoke:
        selected = [ids[:input_tokens] for ids in selected]
    if any(len(ids) != input_tokens or any(type(token) is not int or token < 0 for token in ids)
           for ids in selected):
        raise ValueError(f"每条输入都必须是 {input_tokens} 个非负整数 Token ID")
    digest = hashlib.sha256(json.dumps(selected, separators=(",", ":")).encode()).hexdigest()
    return selected, hashlib.sha256(raw).hexdigest(), digest


def run_ampere(prompts, batch: int, output_tokens: int, repeats: int):
    """模型和物理 KV 池只初始化一次；每个样本仅提交新请求。"""
    import torch
    from ampere_kv.runner import MODEL_REVISION as RUNNER_REVISION, load_model_and_tokenizer
    from ampere_kv.scheduler import FINISHED, Scheduler

    if RUNNER_REVISION != MODEL_REVISION:
        raise RuntimeError("外部基线的模型 revision 与引擎锁定值不同")
    model, _ = load_model_and_tokenizer()
    gpu_inputs = [torch.tensor(ids, device="cuda", dtype=torch.long)[None, :] for ids in prompts]
    budgets = [(len(ids) + output_tokens + BLOCK_SIZE) // BLOCK_SIZE for ids in prompts]
    blocks_per_layer = sum(budgets)
    scheduler = Scheduler(model, blocks_per_layer, max_batch=batch, prefill_chunk_size=0, mixed=False)

    def generate(sample: int):
        requests = [scheduler.submit(f"sample{sample}-r{i}", ids, output_tokens, ignore_eos=True)
                    for i, ids in enumerate(gpu_inputs)]
        while any(request.status != FINISHED for request in requests):
            scheduler.step()
        return [list(request.output_ids) for request in requests]

    generate(-1)  # 一次完整预热；不计入正式样本。
    samples = []
    for index in range(repeats):
        torch.cuda.synchronize()  # 等待预热/上个样本的 GPU 工作，不把它计入本轮。
        start = time.perf_counter()
        outputs = generate(index)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if any(len(ids) != output_tokens for ids in outputs):
            raise RuntimeError("AmpereKV 未生成固定数量的 Token")
        if scheduler.reserved_blocks or any(s._pool.num_free_blocks != blocks_per_layer for s in scheduler.storages):
            raise RuntimeError("AmpereKV 请求结束后没有归还全部 KV 块")
        samples.append({"wall_ms": elapsed_ms, "output_tokens": sum(map(len, outputs))})
    return samples, {
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "kv_dtype": "bfloat16", "kernel": "BF16/V1", "blocks_per_layer": blocks_per_layer,
        "block_size": BLOCK_SIZE, "max_batch": batch, "prefill_chunk_size": 0,
        "mixed": False, "prefix_cache": False, "graph": False,
    }


def run_vllm(prompts, batch: int, output_tokens: int, repeats: int):
    """vLLM 独占另一进程与环境；其阻塞 generate 返回代表输出已在宿主侧。"""
    import torch
    from vllm import LLM, SamplingParams

    if version("vllm") != VLLM_VERSION:
        raise RuntimeError(f"请使用固定的 vLLM {VLLM_VERSION}，当前为 {version('vllm')}")
    config = {
        "model": MODEL_ID, "revision": MODEL_REVISION, "tokenizer_revision": MODEL_REVISION,
        "dtype": "bfloat16", "kv_cache_dtype": "auto", "tensor_parallel_size": 1,
        "max_model_len": 1024, "max_num_seqs": max(4, batch),
        "enable_prefix_caching": False,
    }
    llm = LLM(**config)
    sampling = SamplingParams(temperature=0.0, max_tokens=output_tokens, ignore_eos=True,
                              detokenize=False)
    inputs = [{"prompt_token_ids": ids} for ids in prompts]

    def generate():
        results = llm.generate(inputs, sampling_params=sampling, use_tqdm=False)
        return [list(result.outputs[0].token_ids) for result in results]

    generate()  # 完整预热；前缀缓存关闭，不能复用上一轮的 KV。
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        outputs = generate()
        elapsed_ms = (time.perf_counter() - start) * 1000
        if any(len(ids) != output_tokens for ids in outputs):
            raise RuntimeError("vLLM 未生成固定数量的 Token")
        samples.append({"wall_ms": elapsed_ms, "output_tokens": sum(map(len, outputs))})
    return samples, {
        "vllm": version("vllm"), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "requested_config": config, "graph": "默认启用条件未改；以后端启动日志为准",
        "chunked_prefill": "版本默认值；以后端启动日志为准",
        "attention_backend": "自动选择；以后端启动日志为准",
        "kv_budget": "由 vLLM 根据可用显存决定；以后端启动日志为准",
    }


def main():
    parser = argparse.ArgumentParser(description="单进程单引擎的离线整批完成时间基线")
    parser.add_argument("--engine", required=True, choices=("ampere", "vllm"))
    parser.add_argument("--input", required=True, type=Path, help="私有 JSON：{'prompts': [[Token ID...], ...]}")
    parser.add_argument("--output", required=True, type=Path, help="仓库外的私有结果 JSON 路径")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--batch", type=int, choices=(1, 4), required=True)
    parser.add_argument("--smoke", action="store_true", help="只取第一条输入前16个 Token，生成4个，测1次")
    args = parser.parse_args()
    if args.smoke and args.batch != 1:
        parser.error("烟测只支持 B=1")
    repo = Path(__file__).resolve().parents[1]
    if args.output.resolve().is_relative_to(repo):
        parser.error("私有结果必须写到仓库外，不能成为待提交文件")

    input_tokens, output_tokens, repeats = (16, 4, 1) if args.smoke else (512, 32, 5)
    prompts, file_hash, selected_hash = load_inputs(args.input, args.batch, input_tokens, args.smoke)
    if args.engine == "ampere":
        samples, engine_config = run_ampere(prompts, args.batch, output_tokens, repeats)
    else:
        samples, engine_config = run_vllm(prompts, args.batch, output_tokens, repeats)

    # 只记录摘要与数量，不保存输入 ID、生成 ID、文本或个人路径。
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        text=True).strip()
    times = [sample["wall_ms"] for sample in samples]
    for sample in samples:
        sample["output_tokens_per_s"] = sample["output_tokens"] * 1000 / sample["wall_ms"]
    result = {
        "run_id": args.run_id, "commit": commit, "engine": args.engine,
        "model": MODEL_ID, "revision": MODEL_REVISION, "gpu": gpu,
        "python": sys.version.split()[0], "input_file_sha256": file_hash,
        "selected_token_ids_sha256": selected_hash, "batch": args.batch,
        "input_lengths": [len(ids) for ids in prompts], "output_limit": output_tokens,
        "workload": "offline_fixed_tokens_ignore_eos_no_prefix_reuse",
        "timing": "warmup_excluded_wall_submit_to_all_host_ids_ready",
        "engine_config": engine_config, "samples": samples,
        "median_wall_ms": statistics.median(times),
        "range_wall_ms": max(times) - min(times),
        "median_output_tokens_per_s": statistics.median(s["output_tokens_per_s"] for s in samples),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[完成] {args.engine} B={args.batch} 输入={input_tokens} 输出={output_tokens} "
          f"样本={[round(x, 3) for x in times]} ms 中位数={result['median_wall_ms']:.3f} ms "
          f"吞吐中位数={result['median_output_tokens_per_s']:.3f} Token/s")
    print(f"[指纹] 输入文件SHA256={file_hash}，选中Token SHA256={selected_hash}，提交={commit[:7]}")
    print("[边界] 离线整批完成与输出吞吐；不是 TTFT、TPOT、服务吞吐、P99 或 SLO")


if __name__ == "__main__":
    main()
