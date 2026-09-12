# AmpereKV Engine

面向 NVIDIA Ampere `sm_86`、Qwen3-8B 的最小单卡推理引擎，独立实现，
不是 nano-vLLM、vLLM 或其他推理框架的 Fork。

## 当前范围

- 单请求、无填充输入；BF16 权重；自建 Prefill、Decode 和贪心生成。
- 连续/分页 KV、共享物理存储、BF16 CUDA Paged Decode。
- 每 Token、每 KV 头的 INT8 KV 与独立 FP16 K/V scale；CUDA 内融合反量化。
- 权重与分词器使用固定 Revision，只从本地缓存加载；底层线性计算使用 PyTorch。

BF16 模型对照和 INT8 六个小型 CUDA 场景已有 RTX 3090 验证结果。
第一层真实 INT8 Attention 检查入口已实现，等待 GPU 验证；不代表 INT8 完整模型、
长上下文或性能已通过。当前不提供多请求调度、在线服务或通用模型适配。

## 运行

在云端已激活的环境中执行；需要 CUDA PyTorch、支持 Qwen3 的 Transformers、
Accelerate 和已缓存的固定版本模型。输入文本不写入文件。

```bash
# CUDA 源码变化后构建；纯 Python 修改无需重新构建。
make build
make smoke

# 当前 INT8 算子：六个小型数值对照场景。
python -m ampere_kv.paged_decode

# 最新检查：第一层一次真实 INT8 Decode Attention。
python -m ampere_kv.runner --mode int8-check --cache paged

# 保留的整体回归：自建 BF16 与 HF；CUDA Decode 与自建 SDPA。
python -m ampere_kv.runner --mode check --cache paged
python -m ampere_kv.runner --mode cuda-check --cache paged

# 普通 BF16 生成，以及分页 SDPA/CUDA 两条路径的单请求测量。
python -m ampere_kv.runner --mode generate --cache paged
python -m ampere_kv.runner --mode benchmark --cache paged --repeats 3

# BF16 算子调用基线，不测 INT8 性能。
python -m ampere_kv.paged_decode --benchmark
```

旧版 reference 演示和 block_pool、kv_cache、paged_cache、quantization 的模块自检入口
已删除；核心存储与计算函数保留。历史验证代码可从 Git 历史查看。

## 性能口径

模型和输入已就绪的单请求 Engine-only 墙钟时间，包含 KV 分配、Prefill、Decode、
选词与必要同步；排除模型加载、分词、HF 对照、文本解码、打印及结束时缓存归还。
输出最多 32 Token，遇到 EOS 提前停止。

- TTFT：开始到首个 Token ID 在 CPU 可用。
- TPOT：首个到最后一个 Token 的时间除以后续 Token 数；只输出一个时无定义。
- 输出吞吐：输出 Token 数除以请求生成总时间，不是并发服务吞吐。
- allocated 为活跃张量占用，reserved 为分配器保留空间，峰值包含模型权重。

少量重复中位数不代表 P95/P99；短期显存回到基线不证明长期无泄漏。
算子调用基线包含扩展调用、输出分配和检查同步，不是纯内核计时。
