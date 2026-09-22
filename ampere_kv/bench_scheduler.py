"""受控基线：纯 Decode 吞吐、整段/分块 Prefill 干扰；合成输入与忽略 EOS 只用于计时。

包含 KV 追加、元数据上传和参考读回；轮次耗时不是逐请求 Token 间隔，更不能除以 B 当作 TPOT。
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


def prepare(model, prompts, max_new: int, capacity: int, batch: int, scheduler_type=Scheduler, **kwargs):
    """计时外准备：提交全部请求并跑完各自的完整 Prefill（调度器每轮至多接入一个）。

    `kwargs` 只用来透传构造开关（例如实验侧的 `mixed=True`）；块长由调用方在准备之后设置，
    因为既有请求一律整段 Prefill。混合与分离在准备阶段的每轮 Decode 数相同，补偿公式不变。
    """
    scheduler = scheduler_type(model, capacity, max_batch=batch, **kwargs)
    # 依次准入时前面的请求已经 Decode；补偿这部分输出，计时开始后每条都剩 max_new-1 步。
    requests = [scheduler.submit(f"r{index}", prompt, max_new + len(prompts) - 1 - index, ignore_eos=True)
                for index, prompt in enumerate(prompts)]
    for _ in range(len(prompts)):
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
    tokens = max(tokens, batch + 1)  # 所有请求接入前不能已有请求退场，否则未真正验证该 B。
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
        print(f"[实验A] B={batch}，计时起始各请求KV长度={[history + batch - 1 - i for i in range(batch)]}")
        max_new = steps + 1  # 首 Token 来自 Prefill，之后正好 steps 个纯 Decode 轮
        capacity = sum(budget(history, max_new + batch - 1 - i) for i in range(batch))
        if batch > 4:
            check_batched_result(model, prompts, capacity)
        # B=1 时逐条与批量是同一条内核调用路径，只测一次，不制造两条相同曲线。
        paths = (("逐条", 1), ("批量", batch)) if batch > 1 else (("B=1", 1),)
        for label, size in paths:
            medians, spreads, throughputs, sizes_seen = [], [], [], []
            for _ in range(repeats):
                scheduler, _ = prepare(model, prompts, max_new, capacity, size)
                walls, groups = timed_rounds(scheduler, steps)
                assert all(sum(group) == batch for group in groups), "计时内有请求提前退出，工作量不固定"
                medians.append(statistics.median(walls))
                spreads.append(max(walls) - min(walls))
                throughputs.append(sum(sum(group) for group in groups) / (sum(walls) / 1000))
                sizes_seen.append(max(max(group) for group in groups))
            print(f"[实验A] B={batch}，{label}：每轮墙钟中位数={statistics.median(medians):.3f} ms"
                  f"（{repeats} 次重复的中位数极差={max(medians) - min(medians):.3f}，"
                  f"单次内轮间极差中位数={statistics.median(spreads):.3f}），"
                  f"实际批量={sorted(set(sizes_seen))}，吞吐中位数={statistics.median(throughputs):.1f} Token/s")
        if batch > 1:
            print(f"[实验A] B={batch}：报告轮次墙钟；轮次 ÷ B 不是用户 TPOT")


class TimedScheduler(Scheduler):
    """仅实验使用：在 CPU 已拿到 Token 时记时，不给正常调度器增加逐步历史。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ready = {}

    def _commit(self, request, token):
        now = time.perf_counter()
        super()._commit(request, token)
        self.ready.setdefault(request.request_id, []).append(now)


def bench_insert(model, tokenizer, history: int, steps: int, insert_at: int,
                 insert_histories: tuple[int, ...], repeats: int, active: int = 4,
                 chunks: tuple[int, ...] = (0, 128), executions: tuple[str, ...] = ("separate",)) -> None:
    """同一负载配对比较整段/分块与分离/混合：原请求输出间隔、新请求 TTFT、全部请求完成时间。

    `executions` 默认只有 `separate`，即原默认路径；加 `mixed` 时两组在同一进程内交替跑同一
    负载。混合路径把一次块与本轮 Decode 合成一次前向，首个受影响的 Token 仍按真实取回时刻记录。
    """
    prompts = make_prompts(tokenizer, active, history)
    max_new = steps + 1
    base_capacity = sum(budget(history, max_new + active - 1 - i) for i in range(active))
    variants = [(chunk, execution) for chunk in chunks for execution in executions]
    for insert_history in insert_histories:
        extra = make_prompts(tokenizer, 1, insert_history)[0]
        capacity = base_capacity + budget(insert_history, 8)
        samples = {variant: [] for variant in variants}
        # 每种组合一轮预热；正式重复交替顺序，避免始终先测同一种。
        for run in range(-1, repeats):
            for chunk, execution in (variants if run % 2 == 0 else variants[::-1]):
                scheduler, requests = prepare(model, prompts, max_new, capacity, active, TimedScheduler,
                                              mixed=execution == "mixed")
                scheduler.prefill_chunk_size = chunk
                for key in scheduler.ready:
                    scheduler.ready[key] = scheduler.ready[key][-1:]
                started = time.perf_counter()
                round_index = 0
                while scheduler.running or scheduler.prefilling is not None or scheduler.waiting:
                    if round_index == insert_at:
                        submitted = time.perf_counter()
                        scheduler.submit("late", extra, 8, ignore_eos=True)
                    scheduler.step()
                    round_index += 1
                total = (time.perf_counter() - started) * 1000
                before, after = [], []
                for request in requests:
                    ready = scheduler.ready[request.request_id]
                    for left, right in zip(ready, ready[1:]):
                        (after if right >= submitted else before).append((right - left) * 1000)
                ttft = (scheduler.ready["late"][0] - submitted) * 1000
                assert scheduler.reserved_blocks == 0
                if run >= 0:
                    samples[(chunk, execution)].append(
                        (statistics.median(before), max(after), ttft, total,
                         scheduler.mixed_rounds, scheduler.fallback_rounds))
        print(f"[实验B] 活动={active}，历史={history}，插入轮={insert_at}，新请求输入={insert_history} Token")
        for (chunk, execution), rows in samples.items():
            print(f"  execution={execution}，Chunk={chunk}（0=整段；后续块含分页读回与显式掩码）")
            for index, name in enumerate(("插入前Token间隔中位数", "插入后最大Token间隔", "新请求TTFT",
                                          "全部请求完成时间")):
                values = [row[index] for row in rows]
                print(f"  {name}：{[round(v, 2) for v in values]} ms；中位数={statistics.median(values):.2f}，"
                      f"极差={max(values) - min(values):.2f}")
            print(f"  调度轮计数：混合轮={[row[4] for row in rows]}，回退轮={[row[5] for row in rows]}"
                  "（混合轮数为 0 说明开关没生效，该组不能算混合结果）")
        print("  Token间隔按实际选词就绪时间计算；少量重复不代表 P99、SLO 或 Goodput")


def calibrate_sample(model, tokenizer, batch: int, history: int, processed: int | None, chunk: int):
    """一个格子的一个样本：返回 (被撑长的输出间隔, 块所在轮墙钟, 下一轮墙钟)，单位 ms。

    `processed` 是待插入请求在计时开始前已处理的 Token 数，None 表示这一格不插入任何 Prefill，
    用来拿 Decode-only 对照。待插入 Prompt 长度恒为 `processed + chunk + 1`，所以计时窗口里跑的
    那个块**永远不是最后一块**：值不含最终选词，不能当完整 TTFT。`processed=0` 走首块路径
    （不读回历史），`>0` 走分页读回加显式偏移掩码那条路径，两类成本不同，必须分开记录。
    """
    prompts = make_prompts(tokenizer, batch, history)
    max_new = 8  # 只要保证窗口里既有请求都不退场；输出本身不用于质量评价。
    capacity = sum(budget(history, max_new + batch - 1 - i) for i in range(batch))
    pending = None
    if processed is not None:
        total = processed + chunk + 1
        pending = make_prompts(tokenizer, 1, total)[0]
        capacity += budget(total, 8)  # 插入请求另占一份预算，否则块池容量被低估。
    scheduler, requests = prepare(model, prompts, max_new, capacity, batch, TimedScheduler)
    if pending is not None:
        scheduler.submit("pending", pending, 8, ignore_eos=True)
        if processed:
            # 计时外把待插入请求推进恰好 processed 个 Token；它必须还留在 Prefill 槽位里，
            # 否则下面测到的就是"最终块 + 选词"，不是中间块。
            scheduler.prefill_chunk_size = processed
            scheduler._admit()
            scheduler._advance(scheduler.prefilling, prefill=True)
            assert scheduler.prefilling is not None, "待插入请求已提前选词，测的不是中间块"
    scheduler.prefill_chunk_size = chunk if processed is not None else 0
    for key in scheduler.ready:
        scheduler.ready[key] = scheduler.ready[key][-1:]  # 只留"上一 Token 就绪"作为窗口起点。
    started = time.perf_counter()
    scheduler.step()  # 轮 R：既有请求 Decode，随后才执行那个 C Token 中间块
    # 中间块不选词，此处可能尚未执行完；两列墙钟只描述宿主调用，不是各阶段GPU耗时。
    block_wall = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    # 只推进下一轮既有请求的 Decode：完整 step 会再处理 pending 的末尾1个Token，
    # 让它选词并进入 running（B变B+1），还把第二次Prefill混进下一轮墙钟。
    sizes = scheduler._decode_round()
    next_wall = (time.perf_counter() - started) * 1000
    assert sum(sizes) == batch and scheduler.running == requests, "计时窗口内既有请求集合或工作量改变"
    if pending is not None:
        assert scheduler.prefilling.caches[0].length == processed + chunk
        assert not scheduler.prefilling.output_ids, "中间块不应产生首Token"
    delay = max((scheduler.ready[request.request_id][-1] - scheduler.ready[request.request_id][-2]) * 1000
                for request in requests)
    # 样本结束才回收，不能为清理再跑一次step，改变刚测量的工作量。
    for request in requests + ([scheduler.prefilling] if scheduler.prefilling is not None else []):
        scheduler._retire(request)
    scheduler.running.clear()
    scheduler.prefilling = None
    assert scheduler.reserved_blocks == 0
    return delay, block_wall, next_wall


def bench_calibrate(model, tokenizer, batches: tuple[int, ...], history: int,
                    processed_values: tuple[int, ...], chunks: tuple[int, ...], samples: int) -> None:
    """G6-D1 筛查矩阵：每个 (B,H,C) 的中间块干扰，除以同 B/L 的 Decode-only 对照判可行性。

    只报原始样本、中位数、最大值与极差，不称 P99/SLO/Goodput；1.25× 是实验阈值，不是线上保证。
    """
    cells = [(batch, None, 0) for batch in batches] + [
        (batch, processed, chunk) for batch in batches for processed in processed_values for chunk in chunks]
    found = {cell: [] for cell in cells}
    print(f"[标定] 活动历史={history}，B={list(batches)}，H={list(processed_values)}，"
          f"C={list(chunks)}，每格有效样本={samples}；另加一轮预热丢弃，块大小固定 16 Token")
    print("[口径] 值取既有请求『上一 Token 就绪 → 下一 Token 就绪』的最大间隔；中间块在该轮 Decode 之后"
          "执行，所以影响落在下一轮。待插入 Prompt 恒为 H+C+1，块永远是中间块，不含最终选词，不是 TTFT。")
    print("[配对] H>0只预填插入请求，不推进旧请求；活动请求准备后的KV长度为L+B-1-i（i从0起），"
          "同B的对照与候选一致；下一轮只测Decode，不执行插入请求的末尾块。")
    for index in range(samples + 1):
        offset = index % len(cells)  # 确定性轮换：每轮从不同格子起步，对照与候选同轮穿插
        for cell in cells[offset:] + cells[:offset]:
            batch, processed, chunk = cell
            sample = calibrate_sample(model, tokenizer, batch, history, processed, chunk)
            if index:
                found[cell].append(sample)
    for batch in batches:
        control = found[(batch, None, 0)]
        base = statistics.median(row[0] for row in control)
        print(f"[标定] B={batch} Decode-only 对照：样本={[round(row[0], 2) for row in control]} ms，"
              f"中位数={base:.2f}，实验预算 1.25×={base * 1.25:.2f}")
        for processed in processed_values:
            for chunk in chunks:
                rows = found[(batch, processed, chunk)]
                delays = [row[0] for row in rows]
                median = statistics.median(delays)
                print(f"[标定] B={batch} H={processed} C={chunk}：样本={[round(v, 2) for v in delays]} ms，"
                      f"中位数={median:.2f}，最大={max(delays):.2f}，极差={max(delays) - min(delays):.2f}，"
                      f"相对基线={median / base:.2f}× → "
                      f"{'块长可行' if median <= base * 1.25 else '超预算'}"
                      f"（宿主调用墙钟中位数，非GPU阶段耗时：块所在轮={statistics.median(row[1] for row in rows):.2f}，"
                      f"下一轮={statistics.median(row[2] for row in rows):.2f} ms）")
    print("[标定] 只给趋势与可行性：候选全部超预算时直接报告无可行块长，不增大系数、不换更松指标、"
          "不拿最小块冒充满足；轮墙钟另列一栏，不替代实际间隔")


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
    parser.add_argument("--only", choices=("decode", "insert", "calibrate"), help="只跑其中一组")
    parser.add_argument("--chunks", type=int, nargs="+", default=(0, 128), help="实验 B 的固定块长，0 为整段")
    parser.add_argument("--pending-history", type=int, nargs="+", default=(0, 512),
                        help="实验 C 待插入请求在计时前已处理的 Token 数，0 表示测首块")
    parser.add_argument("--mixed", action="store_true",
                        help="实验 B 与分离路径在同一进程内配对测混合调度；默认只测分离，行为不变")
    args = parser.parse_args()
    if args.repeats < 1 or args.steps < 2 or not 0 < args.insert_at < args.steps:
        parser.error("--repeats/--steps 必须为正，--insert-at 必须在 1 到 --steps-1 之间")
    if min(args.history, *args.batches, *args.insert_history) < 1 or min(args.chunks) < 0:
        parser.error("输入长度与 Batch 必须为正，Chunk 不能为负")
    if min(args.pending_history) < 0:
        parser.error("已处理 Token 数不能为负")
    # 标定只认正块长：CLI 里的 chunk=0 表示整段 Prefill，不是"本轮不插块"，不能当候选混进矩阵。
    if args.only == "calibrate" and min(args.chunks) < 1:
        parser.error("标定矩阵的 --chunks 必须是正块长；整段对照请继续用 --only insert")
    if args.mixed and args.only in ("decode", "calibrate"):
        parser.error("--mixed 只作用于实验 B（insert）；纯 Decode 与标定矩阵没有混合路径")
    device = torch.cuda.get_device_properties(0)
    print(f"[环境] 设备={device.name}，显存={device.total_memory / 1024**3:.0f} GiB，"
          f"torch={torch.__version__}，代码版本={revision()}")
    model, tokenizer = load_model_and_tokenizer()
    if args.only == "calibrate":
        bench_calibrate(model, tokenizer, tuple(dict.fromkeys(args.batches)), args.history,
                        tuple(dict.fromkeys(args.pending_history)), tuple(dict.fromkeys(args.chunks)),
                        args.repeats)
    if args.only in (None, "decode"):
        bench_decode(model, tokenizer, args.history, args.steps, args.repeats, tuple(args.batches))
    if args.only in (None, "insert"):
        bench_insert(model, tokenizer, args.history, args.steps, args.insert_at,
                     tuple(args.insert_history), args.repeats, chunks=tuple(dict.fromkeys(args.chunks)),
                     executions=("separate", "mixed") if args.mixed else ("separate",))
    print("[完成] 三组都只说明本次硬件与代码版本下的相对关系：不是请求级 SLO、不是多用户压测，"
          "也不覆盖 INT8/V3 多请求或 CUDA Graph")


if __name__ == "__main__":
    main()
