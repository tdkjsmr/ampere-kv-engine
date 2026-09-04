# Qwen3-8B 模型契约

## 唯一来源

configs/model.lock.yaml 是模型身份、Revision、元数据哈希和关键 Shape 的
唯一锁定来源。configs/qwen3_8b_3090.yaml 只保存 AmpereKV 的运行配置，
不得重新定义模型结构。

## v0.1 固定字段

| 字段 | 固定值 |
| --- | ---: |
| 模型类型 | qwen3 |
| 架构 | Qwen3ForCausalLM |
| 隐藏层数 | 36 |
| Hidden Size | 4096 |
| Intermediate Size | 12288 |
| Query Head 数 | 32 |
| KV Head 数 | 8 |
| Head Dimension | 128 |
| 最大位置长度 | 40960 |
| 权重精度 | BF16 |

运行前必须由 ampere_kv.model.contract 校验下载后的 config.json，并校验
锁文件中列出的元数据 SHA256。字段不一致时必须停止，不能自动猜测兼容。

## Tokenizer 与生成约束

- Model Revision 与 Tokenizer Revision 必须相同并固定；
- v0.1 正确性基线关闭 Sampling；
- v0.1 关闭 Thinking 模式，避免输出协议影响首个回归基线；
- Prompt 内容只从 local_private 下的本地忽略文件读取；
- 仓库不保存自然语言 Prompt、访问令牌、模型权重或原始用户输入。

## 更改流程

模型、Revision、Tokenizer 或关键 Shape 任一变化，都必须同时更新锁文件、
元数据哈希、CPU 契约测试、HF Reference 和对应 ADR。不同模型生成的结果
不能写入同一个 Run ID。
