#include <torch/extension.h>

// CUDA 实现在 smoke_cuda.cu 中，这里只声明 Python 需要调用的函数。
at::Tensor smoke_add_cuda(const at::Tensor& left, const at::Tensor& right);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  // G0 只暴露一个最小函数，用于证明 Python 能调用自定义 CUDA Kernel。
  module.def("smoke_add", &smoke_add_cuda, "两个 CUDA Tensor 逐元素相加");
}
