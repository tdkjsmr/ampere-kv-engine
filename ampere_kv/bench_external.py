"""固定 Token ID 的离线整批基线；每个进程只加载一种引擎。"""

import argparse
import hashlib
import json
import os
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


def check_output_counts(outputs, batch: int, output_tokens: int, engine: str):
    """只在一次完整生成结束后核对工作量，不插入逐 Token 检查。"""
    if len(outputs) != batch or any(len(ids) != output_tokens for ids in outputs):
        raise RuntimeError(f"{engine} 返回的请求数或每请求 Token 数与固定工作量不符")


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


def check_rope_frequency(model):
    """计时外核对固定频率与旧公式，覆盖零/非零位置和单/多 Token。"""
    import torch
    from ampere_kv.runner import apply_rope

    dim = model.config.head_dim
    exponent = torch.arange(0, dim, 2, dtype=torch.int64).float() / dim
    previous = (1.0 / (model.config.rope_theta ** exponent)).to(model._ampere_inv_freq.device)
    if (model._ampere_inv_freq.shape != (64,) or model._ampere_inv_freq.dtype != torch.float32
            or not torch.equal(model._ampere_inv_freq, previous)):
        raise AssertionError("固定 RoPE 频率与旧公式不一致")
    for indices in ((0,), (17,), (0, 5, 17)):
        positions = torch.tensor([indices], device=previous.device)
        values = torch.arange(len(indices) * dim, device=previous.device).reshape(1, 1, len(indices), dim)
        query = (values.float() / dim).to(torch.bfloat16)
        key = ((values.float() + 1) / dim).to(torch.bfloat16)
        current = apply_rope(query, key, positions, model.config, model._ampere_inv_freq)
        reference = apply_rope(query, key, positions, model.config, previous)
        if not all(torch.equal(actual, expected) for actual, expected in zip(current, reference)):
            raise AssertionError(f"RoPE 位置 {indices} 的输出与旧频率路径不一致")
    print("[PASS] RoPE 固定频率及位置 0/17/0,5,17 的 BF16 Q/K 输出逐位一致")


def run_ampere(prompts, batch: int, output_tokens: int, repeats: int,
               profile: bool = False, rope_check: bool = False):
    """模型和物理 KV 池只初始化一次；每个样本仅提交新请求。"""
    import torch
    from ampere_kv.runner import MODEL_REVISION as RUNNER_REVISION, load_model_and_tokenizer
    from ampere_kv.scheduler import FINISHED, Scheduler

    if RUNNER_REVISION != MODEL_REVISION:
        raise RuntimeError("外部基线的模型 revision 与引擎锁定值不同")
    model, _ = load_model_and_tokenizer()
    if rope_check:
        check_rope_frequency(model)
    gpu_inputs = [torch.tensor(ids, device="cuda", dtype=torch.long)[None, :] for ids in prompts]
    budgets = [(len(ids) + output_tokens + BLOCK_SIZE) // BLOCK_SIZE for ids in prompts]
    blocks_per_layer = sum(budgets)
    if profile:
        class TimelineScheduler(Scheduler):
            """只给诊断批次添加宿主 NVTX 范围，不改变原调度逻辑。"""

            def step(self):
                self.profile_round += 1
                with torch.cuda.nvtx.range(f"step:round={self.profile_round},running_before={len(self.running)}"):
                    return super().step()

            def _advance(self, request, *, prefill):
                history = request.caches[0].length
                stage = "prefill_request" if prefill else "decode_single"
                tokens = min(self.prefill_chunk_size or request.input_ids.shape[1],
                             request.input_ids.shape[1] - history) if prefill else 1
                with torch.cuda.nvtx.range(f"{stage}:tokens={tokens},history={history}"):
                    return super()._advance(request, prefill=prefill)

            def _decode_batch(self, group):
                with torch.cuda.nvtx.range(f"decode_batch:batch={len(group)}"):
                    return super()._decode_batch(group)

        scheduler_class = TimelineScheduler
    else:
        scheduler_class = Scheduler
    scheduler = scheduler_class(model, blocks_per_layer, max_batch=batch, prefill_chunk_size=0, mixed=False)
    if profile:
        scheduler.profile_round = 0

    def generate(sample: int):
        requests = [scheduler.submit(f"sample{sample}-r{i}", ids, output_tokens, ignore_eos=True)
                    for i, ids in enumerate(gpu_inputs)]
        while any(request.status != FINISHED for request in requests):
            scheduler.step()
        return [list(request.output_ids) for request in requests]

    warmup_outputs = generate(-1)  # 复用完整预热结果，只供计时外诊断。
    check_output_counts(warmup_outputs, batch, output_tokens, "AmpereKV 预热")
    profile_outputs = None
    if profile:
        if scheduler.reserved_blocks or any(s._pool.num_free_blocks != blocks_per_layer for s in scheduler.storages):
            raise RuntimeError("AmpereKV 预热结束后没有归还全部 KV 块")
        scheduler.profile_round = 0  # 采集窗口内的调度轮从 1 重新编号。
        torch.cuda.synchronize()  # 窗口前排空预热工作；不计入这次采集。
        torch.cuda.profiler.start()
        try:
            with torch.cuda.nvtx.range("offline_batch"):
                profile_outputs = generate(0)
        finally:
            try:
                torch.cuda.synchronize()  # 捕获区末尾只等待一次，确保本批 GPU 工作完成。
            finally:
                torch.cuda.profiler.stop()
        check_output_counts(profile_outputs, batch, output_tokens, "AmpereKV 时间线")
        if profile_outputs != warmup_outputs:
            raise RuntimeError("AmpereKV 时间线批次与同进程预热的 Token 序列不同")
        if scheduler.reserved_blocks or any(s._pool.num_free_blocks != blocks_per_layer for s in scheduler.storages):
            raise RuntimeError("AmpereKV 时间线批次结束后没有归还全部 KV 块")
    samples = []
    for index in range(repeats):
        torch.cuda.synchronize()  # 等待预热/上个样本的 GPU 工作，不把它计入本轮。
        start = time.perf_counter()
        outputs = generate(index)
        elapsed_ms = (time.perf_counter() - start) * 1000
        check_output_counts(outputs, batch, output_tokens, "AmpereKV")
        if scheduler.reserved_blocks or any(s._pool.num_free_blocks != blocks_per_layer for s in scheduler.storages):
            raise RuntimeError("AmpereKV 请求结束后没有归还全部 KV 块")
        samples.append({"wall_ms": elapsed_ms, "output_tokens": sum(map(len, outputs))})
    return samples, {
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "kv_dtype": "bfloat16", "kernel": "BF16/V1", "blocks_per_layer": blocks_per_layer,
        "block_size": BLOCK_SIZE, "max_batch": batch, "prefill_chunk_size": 0,
        "mixed": False, "prefix_cache": False, "graph": False,
    }, warmup_outputs, profile_outputs


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
    sampling_config = {"temperature": 0.0, "max_tokens": output_tokens,
                       "ignore_eos": True, "detokenize": False}
    sampling = SamplingParams(**sampling_config)
    inputs = [{"prompt_token_ids": ids} for ids in prompts]

    def generate():
        results = llm.generate(inputs, sampling_params=sampling, use_tqdm=False)
        # vLLM 0.29.0 保证结果顺序与输入一致；不依赖内部 request_id。
        return [list(result.outputs[0].token_ids) for result in results]

    warmup_outputs = generate()  # 完整预热；前缀缓存关闭，不能复用上一轮的 KV。
    check_output_counts(warmup_outputs, batch, output_tokens, "vLLM 预热")
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        outputs = generate()
        elapsed_ms = (time.perf_counter() - start) * 1000
        check_output_counts(outputs, batch, output_tokens, "vLLM")
        samples.append({"wall_ms": elapsed_ms, "output_tokens": sum(map(len, outputs))})
    return samples, {
        "vllm": version("vllm"), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "requested_config": config, "requested_sampling": sampling_config,
        "vllm_use_flashinfer_sampler": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER"),
        "observed_config": {"attention_backend": None, "cudagraph_mode": None,
                            "chunked_prefill": None, "kv_cache_dtype": None,
                            "kv_cache_tokens": None, "kv_cache_blocks": None},
    }, warmup_outputs


def main():
    parser = argparse.ArgumentParser(description="单进程单引擎的离线整批完成时间基线")
    parser.add_argument("--engine", required=True, choices=("ampere", "vllm"))
    parser.add_argument("--input", required=True, type=Path, help="私有 JSON：{'prompts': [[Token ID...], ...]}")
    parser.add_argument("--output", required=True, type=Path, help="仓库外的私有结果 JSON 路径")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--batch", type=int, choices=(1, 4), required=True)
    parser.add_argument("--smoke", action="store_true", help="只取第一条输入前16个 Token，生成4个，测1次")
    parser.add_argument("--evidence-only", action="store_true", help="仅运行一次完整预热并保存输出，不测性能")
    parser.add_argument("--profile", action="store_true", help="仅采集 AmpereKV 预热后一批的 Nsight Systems 时间线")
    parser.add_argument("--rope-check", action="store_true", help="仅在 AmpereKV 证据运行前核对固定 RoPE 频率与输出")
    args = parser.parse_args()
    if args.smoke and args.batch != 1:
        parser.error("烟测只支持 B=1")
    if args.profile and (args.engine != "ampere" or args.smoke or args.evidence_only):
        parser.error("--profile 只支持 AmpereKV 正式负载，不能与 --smoke/--evidence-only 同用")
    if args.rope_check and (args.engine != "ampere" or not args.evidence_only or args.smoke):
        parser.error("--rope-check 只支持 AmpereKV 正式 --evidence-only 运行")
    repo = Path(__file__).resolve().parents[1]
    if args.output.resolve().is_relative_to(repo):
        parser.error("私有结果必须写到仓库外，不能成为待提交文件")

    input_tokens, output_tokens, repeats = (16, 4, 1) if args.smoke else (512, 32, 5)
    if args.evidence_only or args.profile:
        repeats = 0
    prompts, file_hash, selected_hash = load_inputs(args.input, args.batch, input_tokens, args.smoke)
    if args.engine == "ampere":
        samples, engine_config, warmup_outputs, profile_outputs = run_ampere(
            prompts, args.batch, output_tokens, repeats, profile=args.profile, rope_check=args.rope_check)
    else:
        samples, engine_config, warmup_outputs = run_vllm(prompts, args.batch, output_tokens, repeats)
        profile_outputs = None

    # 完整预热输出只写仓库外私有结果；公开日志仍不包含输入或生成 Token ID。
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
        "timing": ("profile_only_no_measurement" if args.profile else
                   "warmup_only_no_measurement" if args.evidence_only else
                   "warmup_excluded_wall_submit_to_all_host_ids_ready"),
        "warmup_output_role": "diagnostic_not_timed",
        "warmup_output_token_ids": [
            {"input_index": index, "token_ids": ids} for index, ids in enumerate(warmup_outputs)
        ],
        "engine_config": engine_config, "samples": samples,
    }
    if args.profile:
        result.update({
            "profile_only": True, "profile_output_role": "diagnostic_not_timed",
            "profile_output_token_ids": [
                {"input_index": index, "token_ids": ids} for index, ids in enumerate(profile_outputs)
            ],
        })
    if times:
        result.update({
            "median_wall_ms": statistics.median(times),
            "range_wall_ms": max(times) - min(times),
            "median_output_tokens_per_s": statistics.median(s["output_tokens_per_s"] for s in samples),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if times:
        print(f"[完成] {args.engine} B={args.batch} 输入={input_tokens} 输出={output_tokens} "
              f"样本={[round(x, 3) for x in times]} ms 中位数={result['median_wall_ms']:.3f} ms "
              f"吞吐中位数={result['median_output_tokens_per_s']:.3f} Token/s")
    elif args.profile:
        print(f"[时间线] {args.engine} B={args.batch} 预热后只采集一批；无性能样本")
    else:
        print(f"[证据] {args.engine} B={args.batch} 预热输出已保存；无计时样本，不报告性能")
    print(f"[诊断] 预热请求数={len(warmup_outputs)}，每请求输出数={[len(ids) for ids in warmup_outputs]}；不属于计时样本")
    print(f"[指纹] 输入文件SHA256={file_hash}，选中Token SHA256={selected_hash}，提交={commit[:7]}")
    if times:
        print("[边界] 离线整批完成与输出吞吐；不是 TTFT、TPOT、服务吞吐、P99 或 SLO")
    elif args.profile:
        print("[边界] 本次仅留 Nsight Systems 时间线与诊断输出，不报告无采样器的正式性能")
    else:
        print("[边界] 本次仅留预热输出证据，未执行性能测量")


if __name__ == "__main__":
    main()
