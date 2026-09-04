# AmpereKV Engine

AmpereKV Engine 是一个面向 NVIDIA RTX 3090 24GB（Ampere，`sm_86`）和 Qwen3-8B BF16 的单 GPU 推理引擎项目。

项目聚焦长上下文在线推理中的 KV Cache 容量、Decode Attention 显存访问、Prefill 与 Decode 相互干扰，以及短 Kernel 启动开销。目标是构建一条具备明确正确性契约、可复现实验配置和分层性能测量的最小推理路径。

> 当前状态：v0.1 基础设施已经进入实现阶段。CPU 契约可在无 CUDA
> 环境中检查；CUDA/Triton Smoke 必须在 RTX 3090 云端完成后才能记为通过。

## 实现方式

AmpereKV Engine 是独立实现，不是 nano-vLLM、vLLM 或其他推理框架的
Fork。项目会参考成熟引擎的公开设计思路，但自行定义模块边界、数据结构、
CUDA 算子、测试和实验记录。Hugging Face/PyTorch 用作正确性参考，
vLLM 只在后续阶段作为明确版本的对照实现。

这条边界意味着：

- 不复制上游仓库的提交历史或直接在其代码上改造；
- 每个核心组件都必须能够独立解释、测试和替换；
- 借鉴外部实现时记录来源和许可证，不把外部代码误写为原创；
- 对照框架只提供参考结果，不进入 AmpereKV 的运行时依赖。

## 核心模块

- Paged KV Cache 与请求生命周期管理；
- 面向 GQA 的 Shared-KV Paged Decode CUDA Kernel；
- BF16 KV 与 Per-token/head 对称 INT8 KV；
- Decode Kernel 内融合 INT8 KV 反量化；
- Triton 实现的 RoPE、量化与 Paged KV 写入融合；
- Continuous Batching；
- 基于离线干扰标定的 SLO 感知 Chunked Prefill；
- 面向稳定 Decode Batch Bucket 的 CUDA Graph；
- Micro Kernel、Engine-only 和 HTTP Serving 三层 Benchmark。

## 设计主线

```text
Qwen3-8B BF16 + RTX 3090 24GB
                ↓
        Paged KV Cache
                ↓
 GQA Shared-KV Paged Decode
                ↓
 INT8 KV + Fused Dequantization
                ↓
 RoPE / Quantization / KV Write Fusion
                ↓
 Continuous Batching + Chunked Prefill
                ↓
      Decode CUDA Graph
                ↓
 Correctness / Latency / Goodput / Memory
```

## 仓库结构

```text
ampere_kv/       推理引擎、缓存、调度、CUDA Graph 与服务逻辑
csrc/            C++/CUDA 自定义算子
triton_kernels/  Triton 融合算子
configs/         模型、Benchmark、工作负载与 SLO 配置
tests/           CPU、GPU、Kernel、集成与端到端测试
benchmarks/      Micro、Engine、Serving 与 vLLM 对照测试
scripts/         环境验证、构建、测试、分析与报告脚本
docs/            架构、正确性、实验方法、优化与限制说明
results/         可追溯的结果摘要与发布数据
third_party/     第三方依赖和来源说明
```

## 正确性与测量原则

- 所有正式结果绑定固定的代码提交、模型版本、软件环境和运行配置；
- CUDA Kernel 必须覆盖边界 Shape、非整块 Context、混合长度和异常输入；
- 性能测量同时报告绝对值、统计分布、工作负载和正确性结果；
- Micro Kernel、Engine-only 与 Serving E2E 数据分别测量，不混用统计范围；
- 保留负收益 Shape、失败实验和适用边界，不只展示最优结果；
- 在真实测量完成前，不发布推测性加速数字。

## 项目边界

当前版本固定为单 GPU、自回归文本生成，并以 Qwen3-8B BF16 作为主模型。以下内容不属于当前核心范围：

- 训练与微调；
- 多 GPU 并行、NCCL 或 RDMA；
- Prefix Cache 与跨请求 KV 复用；
- 完整 Speculative Decoding；
- 多模型、多 LoRA 或视觉语言模型；
- 完整生产级 API 兼容层。

本项目不会预设全面超过成熟推理框架。后续对照实验将使用锁定版本的 Hugging Face/PyTorch 和 vLLM，并明确记录功能、测量范围及工程成熟度差异。

## 当前进度

- [x] 初始化仓库与目录结构；
- [x] 锁定 Qwen3-8B 模型版本与静态模型契约；
- [x] 建立 CPU 契约测试、KV 容量计算和运行清单 Schema；
- [x] 编写面向 sm_86 的 CUDA Extension 与 Triton 最小 Smoke；
- [ ] 在 RTX 3090 云端冻结实际开发环境并通过两个 GPU Smoke；
- [ ] 建立 Hugging Face/PyTorch 正确性基线；
- [ ] 实现最小 BF16 推理路径；
- [ ] 实现 Paged KV Cache 与 CUDA Decode Kernel；
- [ ] 实现 INT8 KV 与融合写入；
- [ ] 实现在线调度、Chunked Prefill 与 CUDA Graph；
- [ ] 完成分层 Benchmark、Profiler 分析和可复现发布。

v0.1 的具体范围、运行命令和未完成验收项见
docs/v0.1_status.md。任何 CUDA、模型推理或性能结论都必须由云端实际结果
支持；仅在本地通过语法检查不代表 GPU 路径可用。

在代码、测试和实测结果达到对应阶段的验收条件后，本 README 将持续补充
构建方法、复现命令、正确性结果和性能报告。
