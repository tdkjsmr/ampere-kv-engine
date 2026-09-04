# ADR-001：v0.1 主模型选择

状态：已接受

日期：2026-09-04

## 决策

v0.1 主线固定使用 Qwen/Qwen3-8B，精度为 BF16，单卡设备为
RTX 3090 24GB，目标 CUDA 架构为 sm_86。模型和 Tokenizer 使用
configs/model.lock.yaml 中记录的同一固定 Revision。

## 原因

Qwen3-8B 是纯文本因果语言模型，模型契约与本项目计划的 GQA Shared-KV
Paged Decode 路线一致。它包含 32 个 Query Head、8 个 KV Head 和
128 维 Head，能够直接体现 GQA 的 KV 共享关系。

当前不把 Qwen3.5-9B 作为主线，因为它同时引入视觉模型配置和混合
Linear Attention，会改变本项目的核心算子问题。当前也不使用
Qwen3.8-27B，因为 27B BF16 权重本身已经明显超过 24GB 单卡容量，
不符合 v0.1 的单张 RTX 3090 约束。

## 影响

- 模型形状由固定配置驱动，代码中不得散落无法追踪的魔法数字；
- 首个 BF16 正确性基线必须使用锁定 Revision；
- 以后更换模型必须新增 ADR，不得悄悄覆盖现有基线；
- 当前结论只约束 v0.1，不宣称这是所有硬件和任务上的最佳模型。
