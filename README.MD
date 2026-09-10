# AmpereKV Engine

AmpereKV Engine 是面向 NVIDIA Ampere `sm_86` 的最小单卡推理引擎。
项目独立实现，不是 nano-vLLM、vLLM 或其他推理框架的 Fork。

## 当前阶段：G2，自建 BF16 连续 KV 最小引擎

当前支持 Qwen3-8B、单请求无填充输入、BF16 权重与连续 KV Cache，
包含自建 Decoder Layers、Prefill、逐 Token Decode 和最多 32 个新 Token 的贪心生成。
遇到 EOS 提前停止。权重加载与分词使用 Transformers，底层计算使用 PyTorch/SDPA。

已在 RTX 3090 上验证 check、generate 和 benchmark 三种模式：
逐步 logits、完整 Token 序列与 HF 对照通过，纯生成输出与对照模式一致。
单请求性能基线已完成 1 次预热和 3 次重复测量；重复输出一致，
各次请求结束后的活跃张量显存均回到预热基线。
这些结果仅覆盖本次输入与环境，不代表多输入回归、生成上限分支或长期无泄漏验证，
也没有与其他框架比较的性能收益结论；G2 尚未完成全部验收。
单层分页 BF16 存储及等头数/GQA Prefill、Decode 已通过 CPU 小张量自检。
Runner 新增分页参考选择，尚待真实模型 GPU 验证；默认仍使用连续缓存。
多请求共享块池、核心 Paged Decode CUDA 算子、INT8 KV 与融合反量化仍是后续工作。

## 运行

需要支持 CUDA 的 PyTorch、支持 Qwen3 的 Transformers，以及已缓存的模型权重。
模型与分词器版本由代码中的固定 Revision 指定，加载仅使用本地缓存。
以下命令在项目目录、已激活的环境中执行；输入文本不写入文件。

```bash
# 正确性：自建路径与 HF 独立缓存路径逐步对照；也是不传参数时的默认模式。
python -m ampere_kv.runner --mode check

# 纯生成：不执行 HF 前向对照，结束后输出 Token IDs 和文本。
python -m ampere_kv.runner --mode generate

# 基础测量：先预热 1 次，再对同一输入测量 3 次；可调整重复次数。
python -m ampere_kv.runner --mode benchmark --repeats 3

# 分页参考对照：固定每块 16 Token；待 GPU 验证，不支持 benchmark。
python -m ampere_kv.runner --mode check --cache paged
```

分页也可配合 `--mode generate`；其读回会复制历史，不代表高性能分页内核。
分页 check 会报告 Decode 是否跨块，未覆盖时应更换输入补验；每层目前独占块池。

先使用相同输入确认 check 通过、generate 的完整 Token ID 列表一致，再测量。
benchmark 检查重复输出是否一致，但不能代替 HF 正确性对照。
三种运行模式共用自建模型级前向和生成循环；不为测量另写一套计算。

## 性能测量口径

这是已加载模型、输入 Token IDs 已在 GPU 上的单请求 Engine-only 墙钟测量，
不是纯 CUDA 内核时间或在线服务指标。

- 起点：排空前序 GPU 工作后、分配本次请求 KV Cache 之前。
- 首 Token 时刻：首个贪心 Token ID 通过 `item()` 在 CPU 可用时。
- 终点：最后一个新 Token ID 在 CPU 可用时；输出数量包含 EOS。
- 包含：KV 分配、Prefill、Decode、选词、必要输入检查和 Python 控制开销。
- 排除：模型加载、分词、输入搬运、HF 对照、文本解码、打印和结束时缓存释放。
- 不强制生成满 32 个 Token；比较结果时必须同时核对输入长度、实际输出长度及 EOS 策略。

设起点为 t0，首 Token 时刻为 t1，最后一个 Token 时刻为 tN，实际输出数为 N：

| 指标 | 定义 |
|---|---|
| TTFT，首 Token 延迟 | t1 - t0 |
| 平均 TPOT，后续 Token 平均耗时 | (tN - t1) / (N - 1)；N=1 时为 N/A |
| 请求生成总耗时 | tN - t0 |
| 单请求输出吞吐 | N / (tN - t0) |

打印每次测量值和各指标的中位数；少量重复不用于宣称 P95、并发服务吞吐或显著性能收益。
当前仍有 Python 调度、层内标量读取和临时张量等成本，测量结果不代表优化后的性能。

每次请求独立创建并释放缓存，不调用 `empty_cache()`。显存输出包括：

- 预热后和各请求结束后的 `memory_allocated`，以及相对预热基线的变化；
- 分配器管理的 `memory_reserved`；
- 每次请求的峰值 `max_memory_allocated`，包含模型等常驻张量。

分配器保留空间不等于活跃张量占用，也不是整卡总显存。
短期请求结束后占用稳定不能证明长期无泄漏；未回落时需结合逐次变化进一步检查。

## 独立自检

```bash
# CPU 缓存与 Attention 接口自检，不加载模型。
python -m ampere_kv.kv_cache

# G0 CUDA Extension Smoke；需要 CUDA Toolkit 与 C++ 编译器。
make build
make smoke
```

模型路径的 Python 修改不需要重新编译上述 Smoke 扩展。
CUDA 编译与模型运行结果以实际 GPU 环境输出为准。
