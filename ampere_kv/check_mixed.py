"""混合前向原型对照：不接调度器，不代表请求TTFT、SLO或完整独立生成通过。"""

import argparse
import statistics
import time

import torch

from ampere_kv.bench_scheduler import make_prompts, revision
from ampere_kv.paged_cache import PagedKVCache
from ampere_kv.runner import (load_model_and_tokenizer, model_forward,
                              model_forward_batched, model_forward_mixed)
from ampere_kv.scheduler import Scheduler


def positions(ids, start):
    """位置由本请求历史决定，不能使用打包后的全局Token下标。"""
    return torch.arange(start, start + ids.shape[1], device=ids.device).unsqueeze(0)


def prepare(model, prompts, pending, history, extra):
    """同一套池中各请求独占块；两条被测路径各自创建一套，不共享已分配块。"""
    capacity = sum((p.shape[1] + extra + 15) // 16 for p in prompts) + (pending.shape[1] + 15) // 16
    owner = Scheduler(model, capacity)
    caches = [[PagedKVCache(storage) for storage in owner.storages] for _ in range(len(prompts) + 1)]
    tokens = []
    for prompt, cache in zip(prompts, caches):
        out = model_forward(model, prompt, positions(prompt, 0), cache, is_prefill=True)
        tokens.append(out[:, 0].argmax(-1))
    if history:
        prefix = pending[:, :history]
        model_forward(model, prefix, positions(prefix, 0), caches[-1], is_prefill=True, output_logits=False)
    return owner, caches, torch.stack(tokens).reshape(-1, 1)


def release(state):
    """检查只放在对照入口；退出时归还所有请求的块。"""
    owner, caches, _ = state
    for request in caches:
        for cache in request:
            cache.release()
    assert all(s._pool.num_free_blocks == owner.blocks_per_layer for s in owner.storages)


def compare(actual, expected, label):
    """跨GEMM形状可能有舍入差异：观测logits，要求本次贪心选择一致，不放宽旧容差。"""
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    delta = actual.float() - expected.float()
    relative = delta.norm() / expected.float().norm().clamp_min(1e-12)
    print(f"[观测] {label}：最大误差={delta.abs().max().item():.6g}，相对L2={relative.item():.6g}")
    got, want = actual.argmax(-1), expected.argmax(-1)
    assert torch.equal(got, want), f"{label}选词分歧：混合={got.tolist()}，分离={want.tolist()}"


@torch.inference_mode()
def run_case(model, tokenizer, batch, history, chunks, *, alternate=False, last_logits=True, decode_history=17):
    """相同初始化后连续追加；旧请求由分离参考选词驱动，两侧始终使用同一输入历史。"""
    prompts = [make_prompts(tokenizer, batch, decode_history + 7 * i)[i] for i in range(batch)]
    pending = make_prompts(tokenizer, 1, history + sum(chunks) + (not last_logits))[0]
    states, outputs, timings = {}, {}, {"分离": [], "混合": []}
    try:
        for name in timings:
            states[name] = prepare(model, prompts, pending, history, len(chunks))
        # 初始化的历史应逐位相同，避免把准备差异算进混合计算误差。
        for left, right in zip(states["分离"][1], states["混合"][1]):
            for a, b in zip(left, right):
                if a.length:
                    for x, y in zip(a.get(), b.get()):
                        torch.testing.assert_close(x, y, rtol=0, atol=0)
        addresses = {name: [(s._key.data_ptr(), s._value.data_ptr()) for s in state[0].storages]
                     for name, state in states.items()}
        decode_ids = states["分离"][2]
        start = history
        for step, chunk in enumerate(chunks):
            ids = pending[:, start:start + chunk]
            final = last_logits and step == len(chunks) - 1
            order = ("混合", "分离") if (step + alternate) % 2 else ("分离", "混合")
            for name in order:
                owner, caches, _ = states[name]
                layer_caches = list(map(list, zip(*caches[:-1])))
                pos = torch.tensor([[c[0].length] for c in caches[:-1]], device="cuda")
                ppos = positions(ids, start)
                torch.cuda.synchronize()
                began = time.perf_counter()
                if name == "混合":
                    out = model_forward_mixed(model, decode_ids, pos, layer_caches,
                                              ids, ppos, caches[-1], output_prefill_logits=final)
                else:
                    decoded = model_forward_batched(model, decode_ids, pos, layer_caches)
                    prefilled = model_forward(model, ids, ppos, caches[-1], is_prefill=True,
                                              output_logits=final)
                    out = decoded, prefilled
                torch.cuda.synchronize()
                timings[name].append((time.perf_counter() - began) * 1000)
                outputs[name] = out
                for i, request in enumerate(caches):
                    length = start + chunk if i == batch else prompts[i].shape[1] + step + 1
                    assert all(cache.length == length for cache in request)
                for layer, storage in enumerate(owner.storages):
                    assert addresses[name][layer] == (storage._key.data_ptr(), storage._value.data_ptr())
                    blocks = [block for request in caches for block in request[layer]._table.block_ids]
                    assert len(blocks) == len(set(blocks)), "请求物理块重叠"
            compare(outputs["混合"][0], outputs["分离"][0], f"B={batch} H={start} C={chunk} Decode")
            if final:
                compare(outputs["混合"][1], outputs["分离"][1], "Prefill末块")
            else:
                assert outputs["混合"][1] is None and outputs["分离"][1] is None
            decode_ids = outputs["分离"][0][:, 0].argmax(-1).unsqueeze(1)
            start += chunk
    finally:
        for state in states.values():
            release(state)
    print(f"[PASS] B={batch} 初始H={history} 块长={chunks}：同历史选词、缓存长度、块隔离、地址与回收通过")
    return {name: sum(values) for name, values in timings.items()}


def main():
    parser = argparse.ArgumentParser(description="BF16混合前向独立原型验证")
    parser.add_argument("--benchmark", action="store_true", help="正确性通过后加测B4、C128中间块")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("重复次数必须为正")
    print(f"[环境] {torch.cuda.get_device_name(0)}，torch={torch.__version__}，提交={revision()}")
    model, tokenizer = load_model_and_tokenizer()
    for batch, history in ((1, 0), (4, 17)):
        run_case(model, tokenizer, batch, history, (16, 16, 1))
    if args.benchmark:
        print("[口径] 两路径相同KV与输入；准备/检查/归还不计时，前向含写入、读回和末尾等待，"
              "不含CPU选词；不是用户输出间隔、TTFT或SLO，活动历史为512/519/526/533。")
        for history in (0, 512):
            rows = []
            for repeat in range(args.repeats + 1):
                row = run_case(model, tokenizer, 4, history, (128,),
                               alternate=bool(repeat % 2), last_logits=False, decode_history=512)
                if repeat:
                    rows.append(row)
            for name in ("分离", "混合"):
                values = [row[name] for row in rows]
                print(f"[基线] H={history} {name}：样本={values} ms，中位数={statistics.median(values):.3f}")
    print("[完成] 独立混合前向原型；未接调度器，非HF独立对照、自由生成或SLO验收")


if __name__ == "__main__":
    main()
