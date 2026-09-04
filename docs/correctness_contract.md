# v0.1 正确性契约

## 分层验收

### 第 1 层：CPU 静态契约

- YAML、JSON Schema 与模型配置能够被解析；
- 固定 Revision、关键 Shape 和元数据哈希一致；
- KV Cache 容量公式对 BF16 与 INT8 候选格式给出确定结果；
- 未登记的功能开关被拒绝。

这一层可以在无 CUDA 的 Windows 环境运行，但不能证明 CUDA 构建成功。

### 第 2 层：GPU Smoke

- CUDA Extension 在 sm_86 上完成构建；
- CUDA 向量加法覆盖非整 Block 长度，结果与 PyTorch 完全一致；
- Triton 向量加法覆盖非整 Program 长度，结果与 PyTorch 完全一致；
- GPU 名称、Compute Capability 和结果写入同一 Run ID。

这一层只能在云端 GPU 环境验收。

### 第 3 层：Hugging Face Reference

- 使用锁定模型 Revision 和 Tokenizer；
- 固定 8 个仅存于本地的测试输入；
- Greedy Decode、BF16、Thinking 关闭；
- 保存首步 Logits 摘要、生成 Token ID 和配置哈希；
- 后续 AmpereKV 结果与同一 Reference Run 比较。

## 失败规则

任何异常、NaN、Shape 不匹配、Revision 漂移或证据缺失都视为失败。
不允许把程序能启动当成正确性通过，也不允许用性能结果覆盖正确性失败。

## 当前不承诺的内容

v0.1 尚不承诺完整文本生成、PagedAttention、INT8 KV、在线调度或性能提升。
这些能力必须在后续 Gate 中分别建立测试与回归阈值。
