"""G6-A 多请求生命周期：一个请求对象、一个 FIFO 调度器，共用每层的分页物理存储。

本轮建立的是多请求执行骨架：每个调度轮次里活动请求各跑一次 Decode，至多接入一个新请求做
完整 Prefill。它**不是** Continuous Batching——没有把多个请求的 Token 合成一次 GPU 批量计算，
吞吐提升也不在本轮结论内。只支持 BF16 KV 与默认 V1 Decode 内核，INT8/V3 多请求放到后面。
"""

import torch

from ampere_kv.paged_cache import PagedKVCache, PagedKVStorage
from ampere_kv.runner import (PAGED_BLOCK_SIZE, encode_prompt, generate_tokens,
                              load_model_and_tokenizer, model_forward)

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

    def __init__(self, model, blocks_per_layer: int, *, block_size: int = PAGED_BLOCK_SIZE, device="cuda"):
        config = model.config
        self.model = model
        self.block_size = block_size
        self.blocks_per_layer = blocks_per_layer
        # 每层一份共享物理张量与块池；请求接入时只新建块表，不再分配或复制张量。
        self.storages = [PagedKVStorage(config.num_key_value_heads, config.head_dim, blocks_per_layer,
                                        block_size, device=device) for _ in model.model.layers]
        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self.completed: list[Request] = []
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

    def _advance(self, request: Request, *, prefill: bool) -> None:
        """用请求自己的历史长度构造位置，跑一次模型前向并收下这轮选出的 Token。"""
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
        token = logits[0, 0].argmax().item()
        request.output_ids.append(token)
        if len(request.output_ids) >= request.max_new_tokens:
            request.stop_reason = "达到生成上限"
        elif token in self.eos_ids:
            request.stop_reason = "EOS"

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
        self.completed.append(request)

    def step(self) -> None:
        """一次调度轮次：活动请求各 Decode 一步 → 回收完成者 → 至多接入一个新请求。"""
        for request in list(self.running):
            self._advance(request, prefill=False)
        for request in [r for r in self.running if r.stop_reason]:
            self.running.remove(request)
            self._retire(request)
        admitted = self._admit()
        if admitted is not None and admitted.stop_reason:
            # 首个 Token 就满足停止条件：直接完成，不强行让它进入 Decode。
            self.running.remove(admitted)
            self._retire(admitted)


def check_lifecycle(texts: tuple[str, ...], limits: tuple[int, ...]) -> None:
    """G6-A 四组验收：交错与独立一致、提前完成互不影响、新请求复用归还的块、容量与回收。"""
    model, tokenizer = load_model_and_tokenizer()
    prompts = [encode_prompt(tokenizer, text) for text in texts]
    budgets = [(p.shape[1] + n + PAGED_BLOCK_SIZE) // PAGED_BLOCK_SIZE for p, n in zip(prompts, limits)]
    # 容量取"r1 加上 r0 与 r2 中较大的预算"：r0 与 r1 都在跑时剩余空间必然小于 r2 的预算，
    # r2 一定要等；r0 一释放，剩余空间又必然不小于 r2。这样用例覆盖不依赖具体 Prompt 长度。
    capacity = max(budgets[0], budgets[2]) + budgets[1]
    scheduler = Scheduler(model, capacity)
    requests = [scheduler.submit(f"r{index}", prompt, limit)
                for index, (prompt, limit) in enumerate(zip(prompts, limits))]
    print(f"[观测] 输入 Token={tuple(p.shape[1] for p in prompts)}，每层预算={budgets}，每层容量={capacity} 块")
    held: dict[str, set[int]] = {}
    peak_waiting = 0
    for _ in range(512):  # 逐个轮次手工推进，才能顺便观察等待峰值与各请求占过的物理块。
        if all(request.status == FINISHED for request in requests):
            break
        scheduler.step()
        peak_waiting = max(peak_waiting, len(scheduler.waiting))
        for request in scheduler.running:
            held[request.request_id] = set(request.caches[0]._table.block_ids)
    else:
        raise RuntimeError("超过最大调度轮次，请求未全部完成；检查预留额度与释放路径")
    for request, prompt, limit in zip(requests, prompts, limits):
        # 同一 BF16/V1 路径分别单独运行，作为交错调度的对照：Token 序列必须完全一致。
        alone = generate_tokens(model, prompt, max_new_tokens=limit, cache_kind="paged", cuda_decode=True)
        assert request.output_ids == alone, f"{request.request_id} 交错与独立不一致 {request.output_ids} != {alone}"
    assert peak_waiting > 0, "第三个请求从未等待，复用与容量用例没有覆盖"
    print(f"[PASS] 交错执行与独立执行序列一致：三个请求分别生成 {limits} 个 Token")
    print(f"[PASS] 提前完成互不影响：r0 停止原因={requests[0].stop_reason}，r1 仍完整生成 {limits[1]} 个 Token")
    # 只要求有交集：空闲块栈是后进先出，r2 接入时先弹到 r0 刚归还的编号，但 r1 之后的增长
    # 也可能拿走其中某个块，所以"子集"太严格。真正证明没读到旧历史的是下面与独立运行一致。
    assert held["r2"] & held["r0"], f"r2 没复用到 r0 归还的块：{sorted(held['r2'])} ∩ {sorted(held['r0'])} 为空"
    print(f"[PASS] 新请求复用归还的块：r2 用到 r0 的块编号 {sorted(held['r2'] & held['r0'])}，序列也与独立运行一致")
    assert scheduler.reserved_blocks == 0 and len(scheduler.completed) == len(requests)
    assert all(storage._pool.num_free_blocks == capacity for storage in scheduler.storages)
    oversized = requests[0].input_ids.new_ones(1, capacity * PAGED_BLOCK_SIZE + 1)
    try:
        scheduler.submit("oversized", oversized, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("预算超过整层容量的请求未被拒绝")
    print(f"[PASS] 容量与回收正确：每层 {capacity} 块，收尾预留额度归零、{len(scheduler.storages)} 层块全部归还")


def main() -> None:
    """G6-A 验收入口：加载一次模型跑四组生命周期检查；不测吞吐，也不声称具备批量 Decode。"""
    texts = ("用一句话解释 KV 缓存。", "列举三种可再生能源。", "为什么天空是蓝色的？")
    check_lifecycle(texts, (4, 8, 2))
    print("[完成] G6-A 多请求生命周期四组验收通过；本轮仍是逐请求前向，未覆盖 GPU 批量 Decode、"
          "Chunked Prefill、CUDA Graph、INT8 或 V3 多请求")


if __name__ == "__main__":
    main()
