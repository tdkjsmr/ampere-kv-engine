# 环境分工与证据边界

## Windows 本地环境

用途：编写代码、静态检查、CPU 单元测试、Git 审计和文档维护。

本地没有 CUDA，因此不能在这里验证 nvcc、CUDA Extension、Triton、
模型推理、GPU 显存占用或性能。Windows 静态检查通过时，只能记录为
Local Static PASS。

## RTX 3090 云端环境

用途：构建 sm_86 CUDA 代码、运行 CPU/GPU 测试、执行 Qwen3-8B BF16
推理和常规 Benchmark。每次正式运行必须先生成 environment.json，并将
结果绑定 Git Commit、模型 Revision 和 Run ID。

该环境不提供可用的 GPU 性能计数器，因此不把 Nsight Compute 报告列为
3090 阶段的通过条件。可以继续使用端到端计时、PyTorch Profiler 和
可用的 Nsight Systems 时间线，但不得伪造 NCU 指标。

## A10 云端环境

用途：在代码和工作负载冻结后，集中完成 Nsight Compute 性能计数器分析。
A10 同属 sm_86，但正式报告仍需记录 GPU 型号，不能把 A10 的绝对性能值
冒充 RTX 3090 结果。

为减少按时计费，NCU 工作负载、命令和 Shape 清单应先在仓库中准备完毕，
再一次性租用、采集、下载报告并释放实例。
