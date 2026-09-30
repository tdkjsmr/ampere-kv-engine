# AmpereKV Engine

AmpereKV 是针对固定版本 Qwen3-8B、单张 RTX 3090（`sm_86`）的最小文本推理引擎。项目自行组织逐层执行、分页 KV 缓存、CUDA Decode 和有限多请求调度；Embedding、线性层与 MLP 等仍调用 PyTorch，不是从零实现全部算子，也不是 vLLM 的 Fork 或通用生产服务。

## 能力与边界

- 单请求入口支持 BF16 Prefill、贪心生成、连续或分页 KV；`runner --mode generate --cache paged` 的 BF16 Decode 走 SDPA。多请求调度默认使用 BF16/V1 分页 CUDA Decode；混合 Prefill/Decode 为可选实验路径。V2 可在首次提交前显式启用固定 B1/B4 Decode CUDA Graph，其他批量回退 eager，不改变默认路径。
- 分页缓存按请求维护独立块表，共享每层物理块池；已分配的块归请求独占，完成、合作式取消或到期后归还。块大小固定为 16 Token。
- INT8 仅量化 KV，不量化模型权重：K 每 32 维一组 scale，V 每 Token、每 KV 头一个 scale。相同物理容量下，BF16/INT8 的已测布局字节比约为 `1.9248`，不是实测最大上下文倍率；INT8 多请求、长期质量和最大容量未验收。
- 不提供跨轮会话历史/KV 保留、前缀复用、通用模型适配、多模态或 HTTP 服务。正式版本 `v2.0.0` 的可选 Graph 不支持 INT8/V3、mixed 或动态批次；首版 `v1.0.0` 不包含该功能。现有对照与性能证据仅覆盖报告列出的设备、提交和负载。

## 工具链

V2 运行源码 `56cb10e` 已在已有 Ubuntu 22.04 / CUDA Toolkit `12.4` 的 RTX 3090（24 GiB，驱动 `595.71.05`）上，用独立源码目录和新 venv 完成构建、冒烟与模型/调度/Graph 回归；正式发布仅追加版本字段和文档整理。实际依赖为 Python `3.12.3`、PyTorch `2.5.1+cu124`、`transformers 4.51.0`、`accelerate 1.6.0`、`huggingface_hub 0.36.2`；源码及扩展均来自新目录。系统 Toolkit 的 `nvcc`、PyTorch 的 CUDA runtime 标记 `cu124` 和 NVIDIA 驱动是三个不同版本字段，不应互相替代。

需要 Git、`make`、可用的 C++/CUDA 编译器和足够空间存放约 16 GB 的模型权重。`setup.py` 只构建本项目扩展，**不自动安装所有运行依赖**。以下命令供 CUDA 12.4 云端环境依次执行；若使用其他环境，应先确认与 PyTorch 扩展构建兼容：

```bash
git clone https://github.com/tdkjsmr/ampere-kv-engine.git
cd ampere-kv-engine
git switch --detach v2.0.0  # 固定正式版本；首版请改为 git switch --detach v1.0.0
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip 'setuptools>=69' wheel
python -m pip install 'torch==2.5.1' --index-url https://download.pytorch.org/whl/cu124
python -m pip install 'transformers==4.51.0' 'accelerate==1.6.0' 'huggingface_hub[cli]==0.36.2'
export OMP_NUM_THREADS=1
export HF_HOME=/path/to/persistent/huggingface  # 换成有足够空间的持久化目录；运行时保持一致
hf download Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218
make build
make smoke
python -m ampere_kv.runner --mode generate --cache paged
```

下载模型不需要 GPU；已有固定 Revision 缓存时跳过下载。生成命令会在终端询问一段文本。模型和分词器只从 `HF_HOME` 对应的本地缓存加载，因此切换服务器或缓存目录后须先确认文件仍在。本轮复用了已有驱动、Toolkit 和模型缓存，验证范围是隔离 Python 环境复现，尚未覆盖全新系统安装或首次模型下载。更多模式与参数见 `python -m ampere_kv.runner --help`。

[架构说明](docs/ARCHITECTURE.md) · [固定负载测量报告](docs/BENCHMARK.md)

## 阅读与验收入口

阅读运行主线：`runner.py`（模型前向）→ `scheduler.py`（调度）→ `paged_cache.py`（缓存）→ `graph_decode.py`（可选 Graph）。模型诊断在 `check_model.py`，通过 `runner --mode cuda-check/int8-check --cache paged` 选择其中一种模式。

云端回归入口为 `check_scheduler`、`check_chunked`、`check_mixed`、`check_graph_scheduler`，均用 `python -m ampere_kv.<模块名>` 运行；旧 `scheduler`/`graph_decode` 检查命令仍可用。正式整批测量集中在 `bench_external`，算子测量保留 `paged_decode --benchmark`；参数见各自 `--help`。历史阶段性计时脚本已撤下，原始实现与结果按报告中的 Git 提交追溯，不影响生成接口。
