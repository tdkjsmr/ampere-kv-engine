"""真实模型的固定批量 Graph 调度检查；不计时，不替代独立算子对照。"""

import time

import torch

from ampere_kv.graph_decode import check_case
from ampere_kv.runner import PAGED_BLOCK_SIZE, load_model_and_tokenizer
from ampere_kv.scheduler import FINISHED, Scheduler


def prompt(model, length, offset):
    return torch.arange(offset + 1, offset + length + 1, dtype=torch.long,
                        device="cuda").remainder(model.config.vocab_size)[None]


def start_pair(model, lengths, limits, max_batch, max_tokens):
    blocks = sum((length + limit + PAGED_BLOCK_SIZE) // PAGED_BLOCK_SIZE
                 for length, limit in zip(lengths, limits))
    eager = Scheduler(model, blocks, max_batch=max_batch)
    graph = Scheduler(model, blocks, max_batch=max_batch)
    graph.enable_graphs(max_tokens)
    assert graph.reserved_blocks == 0 and all(s._pool.num_free_blocks == blocks for s in graph.storages)
    assert eager.graphs is None and eager.graph_calls == eager.graph_fallbacks == eager.graph_captures == 0
    inputs = [prompt(model, length, index * 97) for index, length in enumerate(lengths)]
    references = [eager.submit(f"r{index}", ids, limit, ignore_eos=True)
                  for index, (ids, limit) in enumerate(zip(inputs, limits))]
    requests = [graph.submit(f"r{index}", ids, limit, ignore_eos=True)
                for index, (ids, limit) in enumerate(zip(inputs, limits))]
    addresses = {name: [(s._key.data_ptr(), s._value.data_ptr()) for s in scheduler.storages]
                 for name, scheduler in (("eager", eager), ("graph", graph))}
    return eager, graph, references, requests, addresses, blocks


def compare(eager, graph, references, requests, addresses):
    for reference, request in zip(references, requests):
        assert (request.output_ids, request.status, request.stop_reason) == (
            reference.output_ids, reference.status, reference.stop_reason), (
                request.request_id, request.output_ids, reference.output_ids,
                request.status, reference.status, request.stop_reason, reference.stop_reason)
        if request.caches:
            for layer, (cache, other) in enumerate(zip(request.caches, reference.caches)):
                assert cache.length == other.length, (request.request_id, layer, cache.length, other.length)
    for name, scheduler in (("eager", eager), ("graph", graph)):
        active = [request for request in requests if request.caches] if name == "graph" else [
            request for request in references if request.caches]
        for layer, storage in enumerate(scheduler.storages):
            owned = [set(request.caches[layer]._table.block_ids) for request in active]
            assert all(not first & second for index, first in enumerate(owned) for second in owned[index + 1:])
            for request in active:
                cache = request.caches[layer]
                assert cache.length == request.caches[0].length
                assert (cache._key.data_ptr(), cache._value.data_ptr()) == addresses[name][layer]


def check_uploaded(graph, before, groups):
    checks = 0
    for batch, captured in graph.graphs.items():
        if captured.replays == before[batch]:
            continue
        # 用例每轮至多四个活动请求；只核对本轮真实命中的那个子批。
        group = next(group for group in groups if len(group) == batch)
        for layer, (device_starts, device_table) in enumerate(captured.metadata):
            starts, rows = device_starts.tolist(), device_table.tolist()
            for index, request in enumerate(group):
                if not request.caches:
                    continue
                cache = request.caches[layer]
                assert starts[index] == cache.length - 1
                assert rows[index][:len(cache._table.block_ids)] == list(cache._table.block_ids)
                checks += 1
    return checks


def drive(model, lengths, limits, max_batch, max_tokens, *, lifecycle=False):
    eager, graph, references, requests, addresses, blocks = start_pair(
        model, lengths, limits, max_batch, max_tokens)
    sizes_seen, boundary, uploads = set(), [], 0
    stopped = added = reused = new_hit = False
    freed = set()
    after_new = None
    for _ in range(64):
        if all(request.status == FINISHED for request in requests):
            break
        before = {batch: captured.replays for batch, captured in graph.graphs.items()}
        before_calls = graph.graph_calls
        running = list(graph.running)
        groups = [running[offset:offset + graph.max_batch]
                  for offset in range(0, len(running), graph.max_batch)]
        starts = [request.caches[0].length for request in running]
        _, graph_sizes = graph.step()
        _, eager_sizes = eager.step()
        assert graph_sizes == eager_sizes
        sizes_seen.update(graph_sizes)
        compare(eager, graph, references, requests, addresses)
        uploads += check_uploaded(graph, before, groups)
        if added and requests[-1] in running and graph.graph_calls > before_calls:
            new_hit = True
        if max_batch == 1 and running:
            action = "Graph" if graph.graph_calls > before_calls else "eager"
            boundary.append((starts[0], action))
        if lifecycle and not stopped and graph.graphs[4].replays:
            freed = set(requests[0].caches[0]._table.block_ids) | set(requests[1].caches[0]._table.block_ids)
            eager.cancel(references[0])
            graph.cancel(requests[0])
            deadline = time.monotonic() - 1
            references[1].deadline = requests[1].deadline = deadline
            stopped = True
        elif lifecycle and stopped and not added and requests[0].status == requests[1].status == FINISHED:
            ids = prompt(model, 17, 701)
            references.append(eager.submit("new", ids, 8, ignore_eos=True))
            requests.append(graph.submit("new", ids, 8, ignore_eos=True))
            after_new = graph.graph_calls
            added = True
        if added and requests[-1].caches:
            reused |= bool(freed & set(requests[-1].caches[0]._table.block_ids))
    else:
        raise AssertionError("请求未在64轮内完成")
    assert uploads > 0, "Graph 真实请求的起点与块表未核对"
    assert graph.reserved_blocks == eager.reserved_blocks == 0
    assert all(s._pool.num_free_blocks == blocks for s in graph.storages + eager.storages)
    assert eager.graph_calls == eager.graph_fallbacks == eager.graph_captures == 0
    if lifecycle:
        assert requests[0].stop_reason == "已取消" and requests[1].stop_reason == "已超时"
        assert added and reused and new_hit and graph.graph_calls > after_new and graph.graph_captures == 2
        print(f"[PASS] Graph后取消/到期、邻座序列与eager一致；新请求复用释放块并再次命中，捕获仍为{graph.graph_captures}")
    elif max_batch == 1:
        assert (max_tokens - 1, "Graph") in boundary and (max_tokens, "eager") in boundary
        assert graph.graph_calls > 0 and graph.graph_fallbacks > 0 and graph.graph_captures == 1
        print(f"[PASS] B1 容量边界：最后命中起点={max_tokens - 1}，下一步回退起点={max_tokens}；"
              f"命中={graph.graph_calls}，回退={graph.graph_fallbacks}，捕获={graph.graph_captures}")
    else:
        assert {1, 2, 3, 4} <= sizes_seen
        assert graph.graphs[1].replays > 0 and graph.graphs[4].replays > 0
        assert graph.graph_fallbacks > 0 and graph.graph_captures == 2
        print(f"[PASS] B4 逐步准入/退场：子批={sorted(sizes_seen)}，B1/B4命中、B2/B3回退；"
              f"命中={graph.graph_calls}，回退={graph.graph_fallbacks}，捕获={graph.graph_captures}")
    print(f"[PASS] 逐请求完整输出、停止原因、36层长度/地址/块隔离与归还一致；块表上传核对={uploads}")


def main():
    model, _ = load_model_and_tokenizer()
    late = Scheduler(model, 2, max_batch=1)
    late.submit("late", prompt(model, 1, 0), 1)
    mixed = Scheduler(model, 4, max_batch=4, mixed=True)
    for scheduler in (late, mixed):
        try:
            scheduler.enable_graphs(16)
        except ValueError:
            pass
        else:
            raise AssertionError("首次提交后或混合模式错误启用了 Graph")
    print("[PASS] 首次提交后与 mixed 模式拒绝启用 Graph；默认路径未捕获")
    check_case(model, [15])
    check_case(model, [15, 16, 63, 64])
    drive(model, [16], [20], 1, 32)
    drive(model, [17] * 4, [8] * 4, 4, 64)
    drive(model, [17] * 4, [8] * 4, 4, 64, lifecycle=True)
    print("[完成] 固定 B1/B4 Graph 调度接入检查；未验证动态批次Graph、INT8/V3、混合轮或性能")


if __name__ == "__main__":
    main()
