"""G6-B 任务二：批量 Decode 收益与 Prefill 插入干扰两组受控基线；只做实验驱动与统计。

不新增 benchmark 基类、配置系统或自动报告框架；调度器只多了一个"忽略 EOS 以固定工作量"的
开关，逐步计时全部由本文件在外部完成，普通生成路径不保存任何逐步历史。
两组实验都在计时外完成 Prefill 与数据准备，计时包含模型前向、KV 追加、块表等元数据上传、
选词与必要同步（都是调度器本来就要做的那些）。
输出不作质量展示：Prompt 由短句重复拼接后截断到固定长度，EOS 被忽略只为固定工作量。
口径红线：**不能把批量轮次耗时除以 B 冒充用户 TPOT**——那是吞吐折算，不是单个请求两次输出
之间的等待时间。每轮墙钟才是间隔，每轮推进的请求数除以它才是吞吐。
"""

import argparse
import statistics
import subprocess
import time
from pathlib import Path

import torch

from ampere_kv.runner import PAGED_BLOCK_SIZE, encode_prompt, generate_tokens, load_model_and_tokenizer
from ampere_kv.scheduler import FINISHED, Scheduler


def budget(prompt_tokens: int, max_new: int) -> int:
    """每层块预算，与 Scheduler.submit 用同一公式，用来把共享池容量算准。"""
    return (prompt_tokens + max_new + PAGED_BLOCK_SIZE) // PAGED_BLOCK_SIZE


def make_prompts(tokenizer, batch: int, history: int) -> list[torch.Tensor]:
    """造 batch 条长度相同、内容不同的 Prompt。

    长度相同才可比；内容不同才能让"串台"暴露在序列上——八条一样的 Prompt 即使读错了别人的
    历史也会得到相同结果。重复拼接后截断会切掉对话模板结尾，所以这些输出只用于计时。
    """
    prompts = []
    for index in range(batch):
        ids = encode_prompt(tokenizer, f"用一句话解释 KV 缓存的第 {index} 种说法。")
        prompts.append(ids.repeat(1, -(-history // ids.shape[1]))[:, :history])
    return prompts


def prepare(model, prompts, max_new: int, capacity: int, batch: int):
    """计时外准备：提交全部请求并跑完各自的完整 Prefill（调度器每轮至多接入一个）。"""
    scheduler = Scheduler(model, capacity, max_batch=batch)
    requests = [scheduler.submit(f"r{index}", prompt, max_new, ignore_eos=True)
                for index, prompt in enumerate(prompts)]
    for _ in range(len(prompts)):
        if not scheduler.waiting:
            break
        scheduler.step()
    assert len(scheduler.running) == len(prompts) and not scheduler.waiting, "Prefill 没在计时外全部完成"
    return scheduler, requests


def timed_rounds(scheduler, rounds: int):
    """跑 rounds 个纯 Decode 轮次，返回每轮墙钟(ms)与每轮的批量分组大小。"""
    walls, groups = [], []
    for _ in range(rounds):
        started = time.perf_counter()
        _, sizes = scheduler.step()
        walls.append((time.perf_counter() - started) * 1000)
        groups.append(sizes)
    return walls, groups


def check_batched_result(model, prompts, capacity: int, tokens: int = 5) -> None:
    """计时外的结果检查：批量路径与单独运行的序列必须一致，否则不进入测量。

    B 大于既有验收覆盖值时没有正确性证据，不能把"B=4 通过"外推过来。
    """
    batch = len(prompts)
    scheduler = Scheduler(model, capacity, max_batch=batch)
    requests = [scheduler.submit(f"c{index}", prompt, tokens, ignore_eos=True)
                for index, prompt in enumerate(prompts)]
    for _ in range(tokens + batch + 1):
        if all(request.status == FINISHED for request in requests):
            break
        scheduler.step()
    for index, (request, prompt) in enumerate(zip(requests, prompts)):
        alone = generate_tokens(model, prompt, max_new_tokens=tokens, cache_kind="paged",
                                cuda_decode=True, ignore_eos=True)
        assert request.output_ids == alone, f"B={batch} 批量与单独运行不一致：r{index}"
    print(f"[前置] B={batch} 计时外结果检查通过：批量与单独运行的 {tokens} 个 Token 序列一致")


def bench_decode(model, tokenizer, history: int, steps: int, repeats: int, batches: tuple[int, ...]) -> None:
    """实验 A：同一批请求，逐条 Decode 与批量 Decode 的每轮墙钟与吞吐。"""
    print(f"[实验A] 历史={history} Token，Decode 步={steps}，重复={repeats}，B={list(batches)}；"
          "Prefill 全在计时外，计时含前向、KV 追加、元数据上传、选词与必要同步")
    for batch in batches:
        prompts = make_prompts(tokenizer, batch, history)
        max_new = steps + 1  # 首 Token 来自 Prefill，之后正好 steps 个纯 Decode 轮
        capacity = sum(budget(history, max_new) for _ in prompts)
        if batch > 4:
            check_batched_result(model, prompts, capacity)
        # B=1 时逐条与批量是同一条内核调用路径，只测一次，不制造两条相同曲线。
        paths = (("逐条", 1), ("批量", batch)) if batch > 1 else (("B=1", 1),)
        for label, size in paths:
            medians, spreads, throughputs, sizes_seen = [], [], [], []
            for _ in range(repeats):
                scheduler, _ = prepare(model, prompts, max_new, capacity, size)
                walls, groups = timed_rounds(scheduler, steps)
                medians.append(statistics.median(walls))
                spreads.append(max(walls) - min(walls))
                throughputs.append(sum(sum(group) for group in groups) / (sum(walls) / 1000))
                sizes_seen.append(max(max(group) for group in groups))
            print(f"[实验A] B={batch}，{label}：每轮墙钟中位数={statistics.median(medians):.3f} ms"
                  f"（{repeats} 次重复的中位数极差={max(medians) - min(medians):.3f}，"
                  f"单次内轮间极差中位数={statistics.median(spreads):.3f}），"
                  f"实际批量={sorted(set(sizes_seen))}，吞吐中位数={statistics.median(throughputs):.1f} Token/s")
        if batch > 1:
            print(f"[实验A] B={batch}：每轮墙钟即每个请求两次输出之间的间隔；"
                  f"批量轮次 ÷ {batch} 只是吞吐折算，不是用户 TPOT")


def bench_insert(model, tokenizer, history: int, steps: int, insert_at: int,
                 insert_histories: tuple[int, ...], repeats: int, active: int = 4) -> None:
    """实验 B：固定 active 个请求在解码，第 insert_at 轮插入一个新请求做完整 Prefill。

    测三件事：新请求从提交到首 Token 的墙钟（含等待）、原有请求插入前后的每轮间隔、
    插入那一轮被拉长多少。容量按"提交后当轮即可接入"配好，所以这里测的是完整 Prefill 本身
    的阻塞，不含排队；这是受控干扰实验，不是在线服务压测，样本少就报原始值与最大值。
    """
    prompts = make_prompts(tokenizer, active, history)
    max_new = steps + 1
    base_capacity = sum(budget(history, max_new) for _ in prompts)
    for insert_history in insert_histories:
        extra = make_prompts(tokenizer, 1, insert_history)[0]
        capacity = base_capacity + budget(insert_history, 8)
        ttfts, baselines, stretched, overall = [], [], [], []
        for _ in range(repeats):
            scheduler, _ = prepare(model, prompts, max_new, capacity, active)
            late = submitted = first_token = None
            walls = []
            for round_index in range(steps):
                if round_index == insert_at:
                    late = scheduler.submit("late", extra, 8, ignore_eos=True)
                    submitted = time.perf_counter()
                started = time.perf_counter()
                scheduler.step()
                now = time.perf_counter()
                walls.append((now - started) * 1000)
                if late is not None and first_token is None and late.output_ids:
                    first_token = now
            assert late is not None and late.output_ids, "插入的新请求没有被接入，容量或准入有问题"
            ttfts.append((first_token - submitted) * 1000)
            baselines.append(statistics.median(walls[:insert_at]))
            stretched.append(max(walls[insert_at:]))
            overall.append(statistics.median(walls))
        print(f"[实验B] 活动={active}，历史={history}，插入轮={insert_at}/{steps}，新请求输入={insert_history} Token")
        print(f"  首 Token 墙钟（提交到首 Token，含等待）={[round(v, 1) for v in ttfts]} ms，"
              f"中位数={statistics.median(ttfts):.1f} ms")
        print(f"  原有请求插入前每轮间隔中位数={[round(v, 2) for v in baselines]} ms；"
              f"插入后最长一轮={[round(v, 1) for v in stretched]} ms")
        print(f"  被拉长倍数={[round(s / b, 1) for s, b in zip(stretched, baselines)]}；"
              f"整段每轮中位数={[round(v, 2) for v in overall]} ms")
        print("  少量样本，只报原始值、中位数与最大值；未做 P99、SLO 或 Goodput 结论")


def revision() -> str:
    """记录被测代码版本；取不到就写 unknown，不让基线因为环境问题跑不起来。"""
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=Path(__file__).resolve().parents[1]).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    """两组基线入口；默认参数即计划书的第一轮组合，可用参数缩小范围复跑。"""
    parser = argparse.ArgumentParser(description="批量 Decode 收益与 Prefill 插入干扰基线")
    parser.add_argument("--history", type=int, default=512, help="活动请求的历史长度（Token）")
    parser.add_argument("--steps", type=int, default=64, help="计时内的纯 Decode 轮数")
    parser.add_argument("--repeats", type=int, default=3, help="每组重复次数，报中位数与极差")
    parser.add_argument("--batches", type=int, nargs="+", default=(1, 2, 4, 8), help="实验 A 的活动请求数")
    parser.add_argument("--insert-at", type=int, default=32, help="实验 B 在第几轮插入新请求")
    parser.add_argument("--insert-history", type=int, nargs="+", default=(512, 2048), help="实验 B 新请求输入长度")
    parser.add_argument("--only", choices=("decode", "insert"), help="只跑其中一组")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("基线需要云端 CUDA GPU 与已编译扩展")
    if args.repeats < 1 or args.steps < 2 or not 0 < args.insert_at < args.steps:
        parser.error("--repeats/--steps 必须为正，--insert-at 必须在 1 到 --steps-1 之间")
    device = torch.cuda.get_device_properties(0)
    print(f"[环境] 设备={device.name}，显存={device.total_memory / 1024**3:.0f} GiB，"
          f"torch={torch.__version__}，代码版本={revision()}")
    model, tokenizer = load_model_and_tokenizer()
    if args.only in (None, "decode"):
        bench_decode(model, tokenizer, args.history, args.steps, args.repeats, tuple(args.batches))
    if args.only in (None, "insert"):
        bench_insert(model, tokenizer, args.history, args.steps, args.insert_at,
                     tuple(args.insert_history), args.repeats)
    print("[完成] 两组基线只说明本次硬件与代码版本下的相对关系：不是请求级 SLO、不是多用户压测，"
          "也不覆盖 INT8/V3 多请求、Chunked Prefill 或 CUDA Graph")


if __name__ == "__main__":
    main()
