# AmpereKV Engine

AmpereKV Engine 是一个面向 NVIDIA Ampere `sm_86` GPU 的最小单卡推理引擎。
项目独立实现，不是 nano-vLLM、vLLM 或其他推理框架的 Fork。

## 当前阶段：G0

G0 只验证一件事：项目中的 PyTorch CUDA Extension 能够为 `sm_86` 编译，
并在 GPU 上正确执行一个最小 CUDA Kernel。

当前仓库不会提前加入模型配置、Paged KV Cache、Attention、Triton、调度器或
Benchmark；这些组件只在进入对应实现阶段时添加。

## 构建与运行

环境需要预先安装支持 CUDA 的 PyTorch、CUDA Toolkit 和 C++ 编译器。

```bash
make build
make smoke
```

Smoke 成功时会打印：

```text
[PASS] CUDA Extension Smoke
```

本地无 CUDA 的环境只负责阅读和修改代码，GPU 编译与运行结果以实际云端输出为准。
