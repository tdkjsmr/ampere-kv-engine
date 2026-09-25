# AmpereKV Engine

AmpereKV 是针对固定版本 Qwen3-8B、单张 RTX 3090（`sm_86`）的最小文本推理引擎。项目自行组织逐层执行、分页 KV 缓存、CUDA Decode 和有限多请求调度；Embedding、线性层与 MLP 等仍调用 PyTorch，不是从零实现全部算子，也不是 vLLM 的 Fork 或通用生产服务。

## 能力与边界

- 单请求入口支持 BF16 Prefill、贪心生成、连续或分页 KV；`runner --mode generate --cache paged` 的 BF16 Decode 走 SDPA。多请求调度默认使用 BF16/V1 分页 CUDA Decode；混合 Prefill/Decode 为可选实验路径。
- 分页缓存按请求维护独立块表，共享每层物理块池；已分配的块归请求独占，完成、合作式取消或到期后归还。块大小固定为 16 Token。
- INT8 仅量化 KV，不量化模型权重：K 每 32 维一组 scale，V 每 Token、每 KV 头一个 scale。相同物理容量下，BF16/INT8 的已测布局字节比约为 `1.9248`，不是实测最大上下文倍率；INT8 多请求、长期质量和最大容量未验收。
- 不提供跨轮会话历史/KV 保留、前缀复用、通用模型适配、多模态、HTTP 服务或自建 CUDA Graph。现有对照与性能证据仅覆盖报告列出的设备、提交和负载。

## 工具链

已有云端证据使用 Ubuntu 22.04、Python `3.12.3`、PyTorch `2.5.1+cu124`、系统 CUDA Toolkit `12.4`、RTX 3090（24 GiB）和驱动 `595.71.05`；模型对照曾使用 `transformers 4.51.0`。`accelerate 1.6.0` 见既有云端安装记录；`huggingface_hub` 的精确版本未留存，依赖组合尚未在全新环境按下列步骤完整复验。系统 Toolkit 的 `nvcc`、PyTorch 的 CUDA runtime 标记 `cu124` 和 NVIDIA 驱动是三个不同版本字段，不应互相替代。

需要 Git、`make`、可用的 C++/CUDA 编译器和足够空间存放约 16 GB 的模型权重。`setup.py` 只构建本项目扩展，**不自动安装所有运行依赖**。以下命令供 CUDA 12.4 云端环境依次执行；若使用其他环境，应先确认与 PyTorch 扩展构建兼容：

```bash
git clone https://github.com/tdkjsmr/ampere-kv-engine.git
cd ampere-kv-engine
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip 'setuptools>=69' wheel
python -m pip install 'torch==2.5.1' --index-url https://download.pytorch.org/whl/cu124
python -m pip install 'transformers==4.51.0' 'accelerate==1.6.0' 'huggingface_hub[cli]'
export OMP_NUM_THREADS=1
export HF_HOME=/path/to/persistent/huggingface  # 换成有足够空间的持久化目录；运行时保持一致
hf download Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218
make build
make smoke
python -m ampere_kv.runner --mode generate --cache paged
```

下载模型不需要 GPU；生成命令会在终端询问一段文本。模型和分词器只从 `HF_HOME` 对应的本地缓存加载固定 Revision，因此切换服务器或缓存目录后须先确认文件仍在。上面的全新环境安装流程尚未端到端实测通过；已有 GPU 正确性与离线性能证据不等于此流程的验收。更多模式与参数见 `python -m ampere_kv.runner --help`。

[架构说明](docs/ARCHITECTURE.md) · [固定负载测量报告](docs/BENCHMARK.md)
