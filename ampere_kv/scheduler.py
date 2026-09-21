"""G6 多请求引擎：一个请求对象、一个 FIFO 调度器，共用每层的分页物理存储。

G6-A 建立生命周期：等待/活动/完成、容量预留、完成即回收、新请求复用归还的块。
G6-B 增加批量 Decode：`max_batch > 1` 时把一组请求的当前 Token 合成一次模型前向，各请求保留
自己的位置、块表与有效长度，历史既不读回连续显存也不跨请求拼接。`max_batch = 1` 时仍是
G6-A 的逐请求路径，走默认 V1 内核，行为与上一节点完全一致。
本轮仍不是完整的 Continuous Batching：Prefill 还是整段一次做完、每轮至多接入一个新请求，
Chunked Prefill 与 CUDA Graph 在 G6-C/G6-E；只支持 BF16 KV，INT8 与 V3 的多请求未接入。
"""

import torch

from ampere_kv.paged_cache import PagedKVCache, PagedKVStorage
from ampere_kv.runner import (PAGED_BLOCK_SIZE, encode_prompt, generate_tokens,
                              load_model_and_tokenizer, model_forward, model_forward_batched)

WAITING, RUNNING, FINISHED = "waiting", "running", "finished"


class Request:
    """一个请求的输入、输出与逐层缓存；状态只分等待/活动/完成三态。

    能推导的量不再存第二份：有效历史长度读自 `caches[0].length`，下一个输入 Token 就是
    `output_ids` 的最后一个。EOS 与达到上限都只写 `stop_reason`，不各建一套状态。
    """

    def __init__(self, request_id: str, input_ids: torch.Tensor, max_new_tokens: int, budget_blocks: int):
        self.request_id = request_id
        self.input_ids = input_ids  # [1, P] CUDA int64；提交后不再修改。
        self.max_new_tokens = max_new_tokens
        # 调度器级预留额度，与"已分配块数""有效长度"是三回事，见 Scheduler.submit 的注释。
        self.budget_blocks = budget_blocks
        self.status = WAITING
        self.caches: list[PagedKVCache] = []
        self.output_ids: list[int] = []
        self.stop_reason = ""


class Scheduler:
    """各层共享一个 PagedKVStorage，请求只持有自己的块表；单线程按 FIFO 推进。

    共享的是空闲块池，不是已经分配给某个请求的块：分配出的物理块仍由所属请求独占，所以不
    需要引用计数、锁或所有权移交。释放只发生在 `_retire` 一处。
    """

    def __init__(self, model, blocks_per_layer: int, *, block_size: int = PAGED_BLOCK_SIZE,
                 device="cuda", max_batch: int = 1):
        config = model.config
        self.model = model
        self.block_size = block_size
        self.device = device
        # max_batch=1 时完全保持 G6-A 的逐请求路径；>1 才把一组请求的当前 Token 合成一次前向。
        self.max_batch = max_batch
        self.blocks_per_layer = blocks_per_layer
        # 每层一份共享物理张量与块池；请求接入时只新建块表，不再分配或复制张量。
        self.storages = [PagedKVStorage(config.num_key_value_heads, config.head_dim, blocks_per_layer,
                                        block_size, device=device) for _ in model.model.layers]
        self.waiting: list[Request] = []
        self.running: list[Request] = []
        # 只保留"还没结束"的请求。完成的请求由 step() 交回调用方，调度器不长期积累历史，
        # 否则长时间运行会让这两个列表无上限增长，而引擎本身并不需要它们。
        # 各层需求相同，所以一个计数就是每层已预留的块数之和。
        self.reserved_blocks = 0
        eos = model.generation_config.eos_token_id
        self.eos_ids = {eos} if isinstance(eos, int) else set(eos or [])

    def submit(self, request_id: str, input_ids: torch.Tensor, max_new_tokens: int) -> Request:
        """边界校验集中在提交时；内部模型调用不再重复检查自己产出的形状与设备。"""
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("当前只支持单条非空 Token 序列")
        if input_ids.dtype != torch.long or not input_ids.is_cuda:
            raise ValueError("Token IDs 必须是 CUDA 上的 int64 张量")
        if max_new_tokens <= 0:
            raise ValueError("新 Token 上限必须为正数")
        # 预算按每层向上取整，并保守多留一个 Token：最后选出的 Token（含 EOS）不再写入 KV，
        # 但决定它的那次 Decode 已经把该 Token 的 K/V 追加进去了。
        need = (input_ids.shape[1] + max_new_tokens + self.block_size) // self.block_size
        # 只看当前空闲块会高估容量：多个请求会同时消耗尚未兑现的增长空间，最后在 Decode
        # 中途耗尽块池。所以准入用预留额度判断；预算超过整层的请求在提交边界直接拒绝。
        if need > self.blocks_per_layer:
            raise ValueError(f"单请求预算 {need} 块超过每层容量 {self.blocks_per_layer} 块")
        request = Request(request_id, input_ids, max_new_tokens, need)
        self.waiting.append(request)
        return request

    def _commit(self, request: Request, token: int) -> None:
        """收下本轮选出的 Token 并判断停止；EOS 与达到上限只写 stop_reason，不加新状态。"""
        request.output_ids.append(token)
        if len(request.output_ids) >= request.max_new_tokens:
            request.stop_reason = "达到生成上限"
        elif token in self.eos_ids:
            request.stop_reason = "EOS"

    def _advance(self, request: Request, *, prefill: bool) -> None:
        """G6-A 的逐请求路径：一个请求一次前向，Decode 走默认 V1 内核。"""
        start = request.caches[0].length
        if prefill:
            input_ids = request.input_ids
        else:
            input_ids = request.input_ids.new_tensor([[request.output_ids[-1]]])
        positions = torch.arange(start, start + input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        # Prefill 始终走 SDPA；Decode 用默认 V1 内核，与单请求生成路径完全相同。
        logits = model_forward(self.model, input_ids, positions, request.caches,
                               is_prefill=prefill, cuda_decode=not prefill)
        # argmax 之后的 item() 是停止判断本身需要的同步，不属于"为检查而同步"。
        self._commit(request, logits[0, 0].argmax().item())

    def _decode_batch(self, group: list[Request]) -> None:
        """G6-B：一组请求的当前 Token 合成一次模型前向，各请求仍用自己的块表与位置。"""
        tokens = torch.tensor([[request.output_ids[-1]] for request in group],
                              dtype=torch.long, device=self.device)
        # 位置是每个请求各自的绝对位置，不共享——批量不等于对齐历史长度。
        positions = torch.tensor([[request.caches[0].length] for request in group],
                                 dtype=torch.long, device=self.device)
        # 转置成"每层一组请求缓存"：同一层的 B 个缓存指向同一份共享物理存储与块池。
        layer_caches = [[request.caches[layer] for request in group] for layer in range(len(self.storages))]
        logits = model_forward_batched(self.model, tokens, positions, layer_caches)
        # 一次同步取回 B 个 Token：这是各请求停止判断本身需要的，不是为检查而同步。
        for request, token in zip(group, logits[:, 0].argmax(dim=-1).tolist()):
            self._commit(request, token)

    def _decode_round(self) -> list[int]:
        """按 max_batch 分块推进活动请求，返回每一批实际处理的请求数。

        批量大小在真正调用前向的地方记账，不从轮末的活动队列长度反推——同一轮末尾还会接入
        新请求，用队列长度推算会把没参与这次批量 Decode 的请求也算进批里。
        """
        sizes: list[int] = []
        if self.max_batch == 1:
            for request in list(self.running):
                self._advance(request, prefill=False)
                sizes.append(1)
            return sizes
        for offset in range(0, len(self.running), self.max_batch):
            group = self.running[offset:offset + self.max_batch]
            self._decode_batch(group)
            sizes.append(len(group))
        return sizes

    def _admit(self) -> Request | None:
        """每轮最多接入队首一个请求；队首暂时装不下就整队等待，不跳过、也不抢占别人。"""
        if not self.waiting or self.reserved_blocks + self.waiting[0].budget_blocks > self.blocks_per_layer:
            return None
        request = self.waiting.pop(0)
        self.reserved_blocks += request.budget_blocks
        request.caches = [PagedKVCache(storage) for storage in self.storages]
        request.status = RUNNING
        self.running.append(request)
        # 新请求立即执行完整 Prefill 并取得首个 Token；停止条件可能在这一步就满足。
        self._advance(request, prefill=True)
        return request

    def _retire(self, request: Request) -> None:
        """唯一的释放点：归还全部层的块、扣回预留额度，并把请求移出活动集合。"""
        for cache in request.caches:
            cache.release()
        request.caches.clear()  # 丢弃块表引用，请求结束后不可能再被误当成活动请求。
        request.status = FINISHED
        self.reserved_blocks -= request.budget_blocks

    def step(self) -> tuple[list[Request], list[int]]:
        """一次调度轮次：活动请求各 Decode 一步 → 回收完成者 → 至多接入一个新请求。

        返回 (本轮完成的请求, 每批实际处理的请求数)。完成的请求只在这一刻交回调用方，
        调度器自己不留历史；结果需要留存的由调用方持有引用。
        """
        sizes = self._decode_round()
        finished = [r for r in self.running if r.stop_reason]
        for request in finished:
            self.running.remove(request)
            self._retire(request)
        admitted = self._admit()
        if admitted is not None and admitted.stop_reason:
            # 首个 Token 就满足停止条件：直接完成，不强行让它进入 Decode。
            self.running.remove(admitted)
            self._retire(admitted)
            finished.append(admitted)
        return finished, sizes


def _budgets(prompts: list[torch.Tensor], limits: tuple[int, ...]) -> list[int]:
    """按 Scheduler.submit 的同一公式算出每个请求的每层块预算，用例容量由此推得。"""
    return [(p.shape[1] + n + PAGED_BLOCK_SIZE) // PAGED_BLOCK_SIZE for p, n in zip(prompts, limits)]


def _drive(model, prompts, limits, capacity: int, batch: int):
    """提交并跑完一批请求，返回请求列表、调度器、各请求占过的块编号、等待峰值与真实最大成块数。

    完成的请求由 `step()` 交回来，所以这里由调用方自己保存请求引用，调度器不留历史。
    真实最大成块数取每轮 `step()` 报告的批量大小，不从轮末活动队列长度反推。
    """
    scheduler = Scheduler(model, capacity, max_batch=batch)
    requests = [scheduler.submit(f"r{index}", prompt, limit)
                for index, (prompt, limit) in enumerate(zip(prompts, limits))]
    held: dict[str, set[int]] = {}
    peak_waiting = peak_group = 0
    for _ in range(512):
        if all(request.status == FINISHED for request in requests):
            break
        _, sizes = scheduler.step()
        peak_waiting = max(peak_waiting, len(scheduler.waiting))
        peak_group = max(peak_group, max(sizes, default=0))
        for request in scheduler.running:
            held[request.request_id] = set(request.caches[0]._table.block_ids)
    else:
        raise RuntimeError("超过最大调度轮次，请求未全部完成；检查预留额度与释放路径")
    assert scheduler.reserved_blocks == 0, f"收尾预留额度不为零：{scheduler.reserved_blocks}"
    assert all(storage._pool.num_free_blocks == capacity for storage in scheduler.storages), "有请求的块未归还"
    return requests, scheduler, held, peak_waiting, peak_group


def check_lifecycle(model, prompts, limits: tuple[int, ...]) -> None:
    """G6-A 四组验收：交错与独立一致、提前完成互不影响、新请求复用归还的块、容量与回收。"""
    budgets = _budgets(prompts, limits)
    # 容量取"r1 加上 r0 与 r2 中较大的预算"：r0 与 r1 都在跑时剩余空间必然小于 r2 的预算，
    # r2 一定要等；r0 一释放又必然不小于 r2。用例覆盖因此不依赖具体 Prompt 长度。
    capacity = max(budgets[0], budgets[2]) + budgets[1]
    print(f"[观测] 输入 Token={tuple(p.shape[1] for p in prompts)}，每层预算={budgets}，每层容量={capacity} 块")
    requests, scheduler, held, peak_waiting, _ = _drive(model, prompts, limits, capacity, 1)
    for request, prompt, limit in zip(requests, prompts, limits):
        # 同一 BF16/V1 路径分别单独运行，作为交错调度的对照：Token 序列必须完全一致。
        alone = generate_tokens(model, prompt, max_new_tokens=limit, cache_kind="paged", cuda_decode=True)
        assert request.output_ids == alone, f"{request.request_id} 交错与独立不一致 {request.output_ids} != {alone}"
    assert peak_waiting > 0, "第三个请求从未等待，复用与容量用例没有覆盖"
    print(f"[PASS] 交错执行与独立执行序列一致：三个请求分别生成 {limits} 个 Token")
    print(f"[PASS] 提前完成互不影响：r0 停止原因={requests[0].stop_reason}，r1 仍完整生成 {limits[1]} 个")
    # 只要求有交集：块栈后进先出，r2 接入时先弹到 r0 刚归还的编号，但 r1 之后的增长也可能拿走
    # 其中某个块，所以"子集"太严格。证明没读到别人旧历史的是上面与独立运行一致。
    assert held["r2"] & held["r0"], f"r2 没复用到 r0 归还的块：{sorted(held['r2'])} ∩ {sorted(held['r0'])} 为空"
    print(f"[PASS] 新请求复用归还的块：r2 用到 r0 的块编号 {sorted(held['r2'] & held['r0'])}")
    oversized = prompts[0].new_ones(1, capacity * PAGED_BLOCK_SIZE + 1)
    try:
        scheduler.submit("oversized", oversized, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("预算超过整层容量的请求未被拒绝")
    print(f"[PASS] 容量与回收正确：每层 {capacity} 块，收尾预留额度归零、{len(scheduler.storages)} 层块全部归还")


def check_batched(model, prompts, limits: tuple[int, ...], batch: int) -> None:
    """G6-B 验收：批量前向、逐请求前向与单独运行三方给出同样的 Token 序列。

    用例把同一 Prompt 重复拉长，使各请求历史长度不同：跨过 16 Token 的块边界，最长的一条跨
    过分段长度 64 的多个分段；队首请求先退出让批量缩小，队尾请求在中途接入正在跑的批量。
    """
    capacity = sum(_budgets(prompts, limits)[:batch])
    single, _, _, _, single_group = _drive(model, prompts, limits, capacity, 1)
    batched, _, _, peak_waiting, peak_group = _drive(model, prompts, limits, capacity, batch)
    assert single_group == 1 and peak_group == batch, f"真实成块数 {single_group}/{peak_group} 未覆盖 1 与 {batch}"
    assert peak_waiting > 0, "没有请求等待过，中途接入正在跑的批量这一分支没被覆盖"
    for index, prompt in enumerate(prompts):
        alone = generate_tokens(model, prompt, max_new_tokens=limits[index], cache_kind="paged", cuda_decode=True)
        want_single = single[index].output_ids
        want_batch = batched[index].output_ids
        assert alone == want_single == want_batch, (
            f"r{index} 三方不一致：单独={alone}，逐请求={want_single}，批量={want_batch}")
    print(f"[PASS] 批量 Decode 与逐请求、单独运行三方一致：{len(prompts)} 个请求，历史长度 "
          f"{tuple(p.shape[1] for p in prompts)}，生成上限 {limits}，最大成块 {batch}")


def check_batched_operator(starts: tuple[int, ...] = (0, 16, 64)) -> None:
    """批量接口的算子级对照：一次启动放写入后长度为 1、17、65 的三个请求，对齐连续 KV 参考。

    不加载模型，只验新内核的数值与寻址：定宽块表、跨块与跨 64 分段的行、短请求的空分段，
    以及"本步 K/V 是否落在每个请求自己的起点槽位"。块表填充列一律放 -1，内核只要读到填充列
    就会在设备端断言上失败，所以通过本身就证明填充没被读。
    """
    from ampere_kv import _C
    kv_heads, q_heads, block, dim = 2, 4, PAGED_BLOCK_SIZE, 128
    lengths = [start + 1 for start in starts]
    sizes = [(length + block - 1) // block for length in lengths]
    tables, cursor = [], 0
    for size in sizes:
        tables.append(list(range(cursor, cursor + size)))
        cursor += size
    width = max(sizes)
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(0)
    storage = PagedKVStorage(kv_heads, dim, cursor, block, device=device)
    # NaN 哨兵：只有本请求写过的槽位允许变成数字，用它同时证明起点正确和没有多写一格。
    storage._key.fill_(float("nan"))
    storage._value.fill_(float("nan"))
    keys = [torch.randn(1, kv_heads, length, dim, generator=generator, device=device).to(torch.bfloat16)
            for length in lengths]
    values = [torch.randn(1, kv_heads, length, dim, generator=generator, device=device).to(torch.bfloat16)
              for length in lengths]
    for row, key, value, start in zip(tables, keys, values, starts):
        if start:  # 历史用已验证的单请求写入内核放好，批量内核只负责本步那一个 Token。
            _C.bf16_write(key[:, :, :start].contiguous(), value[:, :, :start].contiguous(),
                          storage._key, storage._value, torch.tensor(row, dtype=torch.long, device=device), 0)
    table = torch.tensor([row + [-1] * (width - len(row)) for row in tables], dtype=torch.long, device=device)
    table_starts = torch.tensor(starts, dtype=torch.long, device=device)
    new_key = torch.stack([key[:, :, start:].contiguous() for key, start in zip(keys, starts)])
    new_value = torch.stack([value[:, :, start:].contiguous() for value, start in zip(values, starts)])
    _C.bf16_write_batched(new_key, new_value, storage._key, storage._value, table, table_starts)
    queries = torch.randn(len(starts), q_heads, 1, dim, generator=generator, device=device).to(torch.bfloat16)
    got = _C.paged_decode_batched(queries.contiguous(), storage._key, storage._value, table, table_starts)
    group = q_heads // kv_heads
    for index, (key, value, start) in enumerate(zip(keys, values, starts)):
        # 参考是同一条连续 K/V 上的 FP32 Attention：与内核同一份数据，只换寻址方式。
        want = torch.nn.functional.scaled_dot_product_attention(
            queries[index:index + 1].float().contiguous(), key.float().repeat_interleave(group, dim=1).contiguous(),
            value.float().repeat_interleave(group, dim=1).contiguous(), dropout_p=0.0, is_causal=False,
            scale=dim ** -0.5)
        torch.testing.assert_close(got[index:index + 1].float(), want, rtol=0.01, atol=0.002)
        tail = lengths[index]
        row = tables[index]
        torch.testing.assert_close(storage._key[row[start // block], :, start % block],
                                   key[0, :, start], rtol=0, atol=0)
        assert torch.isnan(storage._key[row[tail // block], :, tail % block]).all(), f"r{index} 多写了一格"
    print(f"[PASS] 批量算子对照：同组长度 {tuple(lengths)} 的数值、新写入位置与未写哨兵全部符合；"
          "块表填充列用 -1，未被读到即断言失败")


def main() -> None:
    """G6 验收入口：先跑不加载模型的批量算子对照，再跑生命周期四组与批量三方一致。"""
    check_batched_operator()
    model, tokenizer = load_model_and_tokenizer()
    prompts = [encode_prompt(tokenizer, text) for text in
               ("用一句话解释 KV 缓存。", "列举三种可再生能源。", "为什么天空是蓝色的？")]
    check_lifecycle(model, prompts, (4, 8, 2))
    # 重复拉长短句制造不同的历史长度：base 约 20 Token，×6 后约 120 Token（跨多个分段）。
    base = prompts[0]
    check_batched(model, [base, base.repeat(1, 3), base.repeat(1, 6), prompts[1].repeat(1, 2), prompts[2]],
                  (3, 6, 4, 8, 2), 4)
    print("[完成] 批量算子对照、G6-A 四组与 G6-B 批量三方一致通过；Prefill 仍是整段一次做完，"
          "未覆盖 Chunked Prefill、CUDA Graph、取消与超时回收，也没有任何吞吐或延迟结论")


if __name__ == "__main__":
    main()
