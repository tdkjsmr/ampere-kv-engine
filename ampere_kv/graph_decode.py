"""固定批量 BF16 Decode Graph 原型；只比较现有批量前向与捕获重放。"""

import argparse
import statistics
import time

import torch

from ampere_kv.paged_cache import PagedKVCache, PagedKVStorage
from ampere_kv.runner import (PAGED_BLOCK_SIZE, load_model_and_tokenizer, model_forward,
                              model_forward_batched, prepare_batched_decode)


def make_state(model, lengths, steps):
    if len(lengths) not in (1, 4) or min(lengths) < 1 or steps < 1:
        raise ValueError("本原型仅支持 B=1/4、非空历史和至少一步 Decode")
    width = max((length + steps + PAGED_BLOCK_SIZE - 1) // PAGED_BLOCK_SIZE for length in lengths)
    blocks = sum((length + steps + PAGED_BLOCK_SIZE - 1) // PAGED_BLOCK_SIZE for length in lengths)
    storages = [PagedKVStorage(model.config.num_key_value_heads, model.config.head_dim,
                               blocks, PAGED_BLOCK_SIZE, device="cuda") for _ in model.model.layers]
    caches = [[PagedKVCache(storage) for _ in lengths] for storage in storages]
    return storages, caches, width


def prefill(model, caches, lengths):
    for request, length in enumerate(lengths):
        ids = torch.arange(1, length + 1, device="cuda", dtype=torch.long).remainder(model.config.vocab_size)[None]
        positions = torch.arange(length, device="cuda", dtype=torch.long)[None]
        model_forward(model, ids, positions, [layer[request] for layer in caches],
                      is_prefill=True, output_logits=False)


def save_prefix(caches):
    return [[cache.get() for cache in layer] for layer in caches]


def restore(storages, caches, prefix):
    for layer in caches:
        for cache in layer:
            cache.release()
    refreshed = [[PagedKVCache(storage) for _ in prefix[0]] for storage in storages]
    for layer, saved in zip(refreshed, prefix):
        for cache, (key, value) in zip(layer, saved):
            cache.append(key, value, fused=True)
    return refreshed


def prepare_layers(caches, width):
    return [prepare_batched_decode(layer, "cuda", width) for layer in caches]


class CapturedDecode:
    """图长期持有静态输入、逐层元数据、输出及捕获时的物理 KV 存储。"""

    def __init__(self, model, caches, width, tokens, positions):
        batch = len(caches[0])
        self.model = model
        self.captured_caches = caches
        self.width = width
        before = (torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
        started = time.perf_counter()
        self.tokens = torch.empty_like(tokens)
        self.positions = torch.empty_like(positions)
        self.metadata = [(torch.empty(batch, dtype=torch.long, device="cuda"),
                          torch.empty(batch, width, dtype=torch.long, device="cuda")) for _ in caches]
        self.tokens.copy_(tokens)
        self.positions.copy_(positions)
        for target, source in zip(self.metadata, prepare_layers(caches, width)):
            target[0].copy_(source[0])
            target[1].copy_(source[1])

        # 侧流预热与捕获只覆盖已经登记的合法位置；宿主长度不能随重放自动前进。
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(3):
                model_forward_batched(model, self.tokens, self.positions, caches, self.metadata)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            self.logits = model_forward_batched(model, self.tokens, self.positions, caches, self.metadata)
        torch.cuda.synchronize()
        self.init_ms = (time.perf_counter() - started) * 1000
        self.extra_bytes = (torch.cuda.memory_allocated() - before[0],
                            torch.cuda.memory_reserved() - before[1])
        self.replays = 0

    def step(self, caches, tokens, positions):
        self.tokens.copy_(tokens)
        self.positions.copy_(positions)
        for target, source in zip(self.metadata, prepare_layers(caches, self.width)):
            target[0].copy_(source[0])
            target[1].copy_(source[1])
        self.graph.replay()
        self.replays += 1
        # 下一次 replay 会覆盖这个输出；调用方必须当步读取或克隆。
        return self.logits


def capture_scheduler_graph(model, storages, batch, max_tokens):
    """在调度器原有物理池上捕获；临时请求只借块，不成为图的所有者。"""
    caches = [[PagedKVCache(storage) for _ in range(batch)] for storage in storages]
    key = torch.zeros(1, storages[0]._key.shape[1], 1, storages[0]._key.shape[3],
                      dtype=torch.bfloat16, device=storages[0]._key.device)
    for layer in caches:
        for cache in layer:
            cache.append(key, key, fused=True)
    tokens = torch.zeros(batch, 1, dtype=torch.long, device=key.device)
    positions = torch.ones_like(tokens)
    try:
        return CapturedDecode(model, caches, max_tokens // PAGED_BLOCK_SIZE, tokens, positions)
    finally:
        torch.cuda.synchronize()
        for layer in caches:
            for cache in layer:
                cache.release()


def inputs(model, lengths, step):
    tokens = torch.tensor([[(11 + 37 * request + step * 13) % model.config.vocab_size]
                           for request in range(len(lengths))], dtype=torch.long, device="cuda")
    positions = torch.tensor([[length + step] for length in lengths], dtype=torch.long, device="cuda")
    return tokens, positions


def check_equal(actual, expected, label):
    if not torch.equal(actual, expected):
        first = (actual != expected).nonzero()[0].tolist()
        error = (actual.float() - expected.float()).abs().max().item()
        raise AssertionError(f"{label} 首个不同索引={first}，最大绝对误差={error:.9g}")


def check_state(paths, addresses, lengths, step):
    reference = paths[0][1]
    for name, caches in paths:
        for layer_index, (layer, ref_layer) in enumerate(zip(caches, reference)):
            ownership = [set(cache._table.block_ids) for cache in layer]
            if any(a & b for index, a in enumerate(ownership) for b in ownership[index + 1:]):
                raise AssertionError(f"{name} 第{layer_index}层出现跨请求块重叠")
            for request, (cache, ref) in enumerate(zip(layer, ref_layer)):
                if cache.length != lengths[request] + step + 1:
                    raise AssertionError(f"{name} 第{layer_index}层请求{request}长度错误")
                if (cache._key.data_ptr(), cache._value.data_ptr()) != addresses[name][layer_index]:
                    raise AssertionError(f"{name} 第{layer_index}层存储地址改变")
                for actual, expected in zip(cache.get(), ref.get()):
                    check_equal(actual, expected, f"{name} 第{layer_index}层请求{request} KV")


def create_paths(model, lengths, steps):
    eager_storage, eager, width = make_state(model, lengths, steps)
    prefill(model, eager, lengths)
    prefix = save_prefix(eager)
    paths = [("旧eager", eager_storage, eager)]
    for name in ("预备eager", "Graph"):
        storage, caches, _ = make_state(model, lengths, steps)
        paths.append((name, storage, restore(storage, caches, prefix)))
    return paths, prefix, width


def run_step(model, name, caches, width, tokens, positions, captured):
    if name == "旧eager":
        return model_forward_batched(model, tokens, positions, caches)
    if name == "预备eager":
        return model_forward_batched(model, tokens, positions, caches, prepare_layers(caches, width))
    return captured.step(caches, tokens, positions)


def close_paths(paths, blocks):
    for name, storages, caches in paths:
        for layer, storage in zip(caches, storages):
            for cache in layer:
                cache.release()
            if storage._pool.num_free_blocks != blocks:
                raise AssertionError(f"{name} 块未全部归还")


def check_case(model, lengths):
    steps = 4
    paths, prefix, width = create_paths(model, lengths, steps)
    tokens, positions = inputs(model, lengths, 0)
    graph = CapturedDecode(model, paths[2][2], width, tokens, positions)
    captured_table = tuple(tuple(cache._table.block_ids) for cache in paths[2][2][0])
    paths[2] = (paths[2][0], paths[2][1], restore(paths[2][1], paths[2][2], prefix))
    addresses = {name: [(layer[0]._key.data_ptr(), layer[0]._value.data_ptr()) for layer in caches]
                 for name, _, caches in paths}
    seen_tables = set()
    for step in range(steps):
        tokens, positions = inputs(model, lengths, step)
        outputs = [(name, run_step(model, name, caches, width, tokens, positions, graph).clone())
                   for name, _, caches in paths]
        for name, output in outputs[1:]:
            check_equal(output, outputs[0][1], f"B={len(lengths)} 第{step + 1}步 {name} logits")
            check_equal(output.argmax(dim=-1), outputs[0][1].argmax(dim=-1), "选词")
        check_state([(name, caches) for name, _, caches in paths], addresses, lengths, step)
        seen_tables.add(tuple(tuple(cache._table.block_ids) for cache in paths[2][2][0]))
    if len(seen_tables) < 2 or all(table == captured_table for table in seen_tables):
        raise AssertionError("Graph 重放未覆盖变化的块表")
    if graph.replays != steps:
        raise AssertionError("Graph 重放次数与真实 Decode 步数不一致")
    close_paths(paths, sum((length + steps + 15) // 16 for length in lengths))
    print(f"[PASS] B={len(lengths)} 历史={tuple(lengths)}：旧/预备eager/Graph 四步 logits、选词、36层KV、地址、块归属与回收严格一致；重放={graph.replays}，块表发生变化")


def benchmark_case(model, batch):
    lengths = [512] * batch
    steps = 31
    paths, prefix, width = create_paths(model, lengths, steps)
    graph = CapturedDecode(model, paths[2][2], width, *inputs(model, lengths, 0))
    paths[2] = (paths[2][0], paths[2][1], restore(paths[2][1], paths[2][2], prefix))
    samples = {name: [] for name, _, _ in paths}
    order = ("旧eager", "预备eager", "Graph")
    for group in range(6):
        for name in order[group % 3:] + order[:group % 3]:
            index = next(i for i, item in enumerate(paths) if item[0] == name)
            path_name, storages, caches = paths[index]
            if group:
                caches = restore(storages, caches, prefix)
                paths[index] = (path_name, storages, caches)
            torch.cuda.synchronize()
            started = time.perf_counter()
            for step in range(steps):
                tokens, positions = inputs(model, lengths, step)
                run_step(model, name, caches, width, tokens, positions, graph)
            torch.cuda.synchronize()
            if group:
                samples[name].append((time.perf_counter() - started) * 1000)
    close_paths(paths, batch * width)
    print(f"[配置] B={batch}，历史=512，Decode=31步，块表宽度={width}，预热=1组，交错测量=5组")
    print(f"[捕获] 初始化={graph.init_ms:.3f} ms，附加allocated={graph.extra_bytes[0]}字节，reserved={graph.extra_bytes[1]}字节")
    for name in order:
        values = [round(value, 3) for value in samples[name]]
        median = statistics.median(samples[name])
        print(f"[基线] {name}：31步样本={values} ms，中位数={median:.3f} ms，平均每步={median / steps:.3f} ms")
    print("[边界] 含宿主元数据准备、输入更新和组末等待；不含Prefill、恢复、捕获与质量检查，不是TTFT/TPOT或服务吞吐")


def main():
    parser = argparse.ArgumentParser(description="固定批量 BF16 Decode CUDA Graph 原型")
    parser.add_argument("--benchmark", action="store_true", help="正确性通过后测量三条路径")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("本入口必须在 CUDA 设备运行")
    model, _ = load_model_and_tokenizer()
    if model.config.head_dim != 128 or PAGED_BLOCK_SIZE != 16:
        raise RuntimeError("本原型仅支持每头128维、块大小16的 BF16 V1")
    with torch.no_grad():
        check_case(model, [15])
        check_case(model, [15, 16, 63, 64])
        if args.benchmark:
            benchmark_case(model, 1)
            benchmark_case(model, 4)
    print("[完成] 固定 B=1/B=4 Graph 原型；未接调度器、INT8/V3、动态批次或服务路径")


if __name__ == "__main__":
    main()
