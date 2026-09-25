# AmpereKV Engine

面向 RTX 3090（`sm_86`）和固定版本 Qwen3-8B 的单卡推理引擎。执行链、分页 KV 缓存、Decode CUDA 算子与请求调度由本项目实现；Embedding、线性层等仍使用 PyTorch。本项目不是 vLLM 或 nano-vLLM 的 Fork。

## 已实现范围

- BF16 权重与 KV：Prefill、贪心 Decode、连续或分页缓存；分页 Decode 默认使用 CUDA V1 算子。
- 多请求：独立块表与共享物理块池、批量 Decode、固定大小 Chunked Prefill；混合 Prefill/Decode 是可选路径，默认关闭。
- 请求生命周期：轮次边界上的合作式取消和到期处理，完成后归还块。
- INT8 KV：K 每 32 维一组 scale，V 每 Token、每 KV 头一个 scale；CUDA 路径融合反量化。相同物理容量下，已测布局字节比约为 BF16/INT8 = 1.9248；这不是实测最大上下文倍率。
- 可选 V3 GQA 共享载入 Decode 内核；多请求调度默认仍使用 BF16/V1。

模型与分词器使用固定 Revision，从本地 Hugging Face 缓存加载。BF16 模型对照、分页缓存、混合调度及有限 INT8 诊断已有 RTX 3090 结果；INT8 的多请求、长期质量与最大容量尚未验收。不提供 CUDA Graph、HTTP 服务、前缀复用或通用模型适配。

## 运行

在已配置模型缓存和 CUDA PyTorch 的云端环境中：

```bash
make build
make smoke
python -m ampere_kv.runner --mode check --cache paged
python -m ampere_kv.runner --mode cuda-check --cache paged
python -m ampere_kv.check_mixed
python -m ampere_kv.runner --mode generate --cache paged
```

CUDA 源码变化后需要重新构建扩展；仅修改 Python 或文档则不需要。`check` 是与 HF 的有界正确性对照，`cuda-check` 检查分页 CUDA Decode，`check_mixed` 检查混合前向与调度，均不是性能测量。单请求计时与 INT8 诊断的完整命令见 `runner.py --help`。

## 测量边界

[外部基线报告](docs/BENCHMARK.md)记录固定 Token ID、锁定版本 vLLM 的离线整批完成时间与输出吞吐。模型加载、分词和预热不计入；AmpereKV 接收预建 CUDA 输入，vLLM 接收宿主 Token ID 列表，两边入口并非完全同构。该结果不是 TTFT、TPOT、在线吞吐、P99 或 SLO。

项目内部 `runner --mode benchmark` 则是另一种单请求口径：从模型与输入就绪开始，包含 KV 分配、选词与必要同步，但不含结束后的块归还。不同口径的数字不直接相除；INT8/BF16 固定步数计时忽略 EOS，不能用作生成质量评估。`allocated` 是活跃张量占用，`reserved` 是分配器保留空间，短期回到基线不证明长期无泄漏。
