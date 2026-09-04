#pragma once

#include <ATen/ATen.h>

// CUDA 实现在 smoke_cuda.cu 中定义，bindings.cpp 只负责算子 Schema
// 和后端注册。把声明集中在此处可以避免后续正式算子重复声明。
at::Tensor smoke_add_cuda(
    const at::Tensor& left,
    const at::Tensor& right);
