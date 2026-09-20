# AmpereKV Engine

面向 NVIDIA Ampere `sm_86`、Qwen3-8B 的最小单卡推理引擎，独立实现，
不是 nano-vLLM、vLLM 或其他推理框架的 Fork。

## 当前范围

- 单请求、无填充输入；BF16 权重；自建 Prefill、Decode 和贪心生成。
- 连续/分页 KV、共享物理存储、BF16 CUDA Paged Decode。
- 每 Token、每 KV 头的 INT8 KV 与独立 FP16 K/V scale；CUDA 内融合反量化。
- 权重与分词器使用固定 Revision，只从本地缓存加载；底层线性计算使用 PyTorch。

BF16 模型对照和 INT8 六个小型 CUDA 场景已有 RTX 3090 验证结果。
K scale已改为每32维一组，V仍每Token每KV头一个scale；新存储CPU检查通过。
分组CUDA读取与第一层真实对照已在临时sm_89实例通过检查，不代表3090新版本或性能验证。
全模型一次INT8/BF16 CUDA Decode已在3090完成运行与缓存检查，选词一致不代表精度验收。
int8-check提供BF16选词驱动的同历史对照，已有短上下文运行结果，不代表完整精度验收。
INT8独立生成已有EOS与输出上限两种短输入运行结果，不代表完整质量验收。
BF16/INT8 CUDA固定长度计时与实际KV字节统计已有3090短输入结果。
单Token融合版本中INT8 TTFT仍较高，TPOT略低；仅三次测量，不宣称稳定加速。
单Token融合写入的六个算子场景与模型对照已有云端通过反馈。
融合写入现扩展到整段Prefill，与Decode共用内核和量化规则；需重新编译，批量版本等待云端验证。
输入须有限且scale可表示，不逐Token扫描NaN/Inf；Prefill Attention仍使用原始BF16 K/V。
模型诊断保留数值对照和长度检查，移除重复地址/页表遍历；CUDA接口仍保留必要类型、形状与越界保护。
当前不提供多请求调度、在线服务或通用模型适配。

## 运行

在云端已激活的环境中执行；需要 CUDA PyTorch、支持 Qwen3 的 Transformers、
Accelerate 和已缓存的固定版本模型。输入文本不写入文件。

```bash
# CUDA 源码变化后构建；纯 Python 修改无需重新构建。
make build
make smoke

# 可选：分组K存储与参考读回，CPU执行，无需GPU。
python -m ampere_kv.paged_cache

# 重新编译后：批量写入/Decode追加检查与BF16驱动的INT8同历史对照。
python -m ampere_kv.paged_decode
python -m ampere_kv.runner --mode int8-check --cache paged

# int8-check只汇总Decode一致率、logits误差及首次分歧；不是INT8独立生成。

# INT8独立选词，最多32个输出Token，遇到自身EOS停止。
python -m ampere_kv.runner --mode generate --cache paged --kv-dtype int8

# 同一输入依次测BF16 CUDA与INT8 CUDA；均忽略EOS，固定32个输出Token。
python -m ampere_kv.runner --mode benchmark --cache paged --kv-dtype int8 --repeats 3

# 保留的整体回归：自建 BF16 与 HF；CUDA Decode 与自建 SDPA。
python -m ampere_kv.runner --mode check --cache paged
python -m ampere_kv.runner --mode cuda-check --cache paged

# 普通 BF16 生成，以及分页 SDPA/CUDA 两条路径的单请求测量。
python -m ampere_kv.runner --mode generate --cache paged
python -m ampere_kv.runner --mode benchmark --cache paged --repeats 3

# BF16 算子调用基线，不测 INT8 性能。
python -m ampere_kv.paged_decode --benchmark
```

旧版 reference 演示与历史CPU自检已删除；paged_cache只保留当前分组K检查。
历史验证代码可从 Git 历史查看。INT8 CUDA检查已恢复，必须重新编译扩展。

## 性能口径

模型和输入已就绪的单请求 Engine-only 墙钟时间，包含 KV 分配、Prefill、Decode、
选词与必要同步；排除模型加载、分词、HF 对照、文本解码、打印及结束时缓存归还。
普通生成最多32 Token，遇到EOS提前停止；benchmark固定32 Token（31次Decode），忽略EOS。
新计时口径不能直接与旧版遇EOS停止的数据混合比较，固定步数输出不用于质量评估。
INT8配对基线包含两边KV追加及INT8量化写入；两边独立选词，Token历史可能不同。
KV容量统计来自实际预留张量，包括INT8数据与FP16 scale，不含权重和页表；不是进程显存节省比例。

- TTFT：开始到首个 Token ID 在 CPU 可用。
- TPOT：首个到最后一个 Token 的时间除以后续 Token 数；只输出一个时无定义。
- 输出吞吐：输出 Token 数除以请求生成总时间，不是并发服务吞吐。
- allocated 为活跃张量占用，reserved 为分配器保留空间，峰值包含模型权重。

少量重复中位数不代表 P95/P99；短期显存回到基线不证明长期无泄漏。
算子调用基线包含扩展调用、输出分配和检查同步，不是纯内核计时。
