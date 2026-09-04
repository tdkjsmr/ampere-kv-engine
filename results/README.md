# 结果目录规则

results 只保存经过筛选、可公开且可追溯的结果摘要。

临时 Run、原始 Token/Logits、模型输出、Profiler 原始报告和可能包含输入
内容的文件不得提交。它们分别保存在被 Git 忽略的 results/runs、
local_private 和 profiles/raw 目录。

公开结果至少必须记录：

- Git Commit；
- 模型 ID 与固定 Revision；
- Run ID 与 UTC 时间；
- GPU 型号、Compute Capability 和显存；
- CUDA、PyTorch、Triton、编译器版本；
- 完整工作负载 Shape 和功能开关；
- 正确性状态、失败条件与统计方法。

任何尚未实测的数据都不能作为性能结论写入发布目录。
