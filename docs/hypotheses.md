# 优化假设登记

本文件只登记可检验假设，不记录尚未测量的加速结论。

| 编号 | 假设 | 需要的证据 | 最早阶段 |
| --- | --- | --- | --- |
| H-001 | GQA 的同组 Query Head 共享 KV 读取可降低重复显存流量 | 结果正确性、DRAM 指标、Kernel 时间 | Paged Decode |
| H-002 | Paged KV 避免连续大块预留后可提高有效 KV 容量 | 容量公式、峰值显存、碎片统计 | KV Manager |
| H-003 | INT8 KV 能降低 Decode 访存，但量化误差与反量化开销可能抵消收益 | Logits 误差、Token 一致率、带宽和时间 | INT8 KV |
| H-004 | 融合 RoPE、量化和 KV 写入可减少短 Kernel 启动与中间写回 | Kernel 数、时间线、端到端延迟 | Fusion |
| H-005 | Chunked Prefill 可改善 Decode 尾延迟，但过小 Chunk 会降低吞吐 | TPOT P50/P99、TTFT、Goodput | Scheduler |
| H-006 | 稳定 Batch Bucket 的 CUDA Graph 可降低 Launch Overhead | CPU/GPU 时间线、TPOT 分布 | CUDA Graph |

每项假设都允许被证伪。负收益 Shape、失败条件和测量噪声必须与正结果一起
保留。只有固定代码、环境、模型和工作负载后的实测结果才能进入发布摘要。
