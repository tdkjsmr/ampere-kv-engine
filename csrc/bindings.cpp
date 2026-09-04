#include <torch/extension.h>
#include <torch/library.h>

#include "ops.h"

namespace {

at::Tensor smoke_add_meta(
    const at::Tensor& left,
    const at::Tensor& right) {
  // Meta 实现不做真实计算，只校验输出的形状与 dtype 推导。
  // 后续 torch.compile、FakeTensor 和 opcheck 都依赖这一契约。
  TORCH_CHECK(
      left.sizes() == right.sizes(),
      "smoke_add: 两个输入的形状必须一致");
  TORCH_CHECK(
      left.scalar_type() == right.scalar_type(),
      "smoke_add: 两个输入的 dtype 必须一致");
  return at::empty_like(left);
}

}  // namespace

TORCH_LIBRARY(ampere_kv, module) {
  // v0.1 只注册最小 Smoke 算子。正式 Paged Decode 的 Schema
  // 必须等 G4 开始前再次冻结，不能在 G0 阶段提前占位实现。
  module.def("smoke_add(Tensor left, Tensor right) -> Tensor");
}

TORCH_LIBRARY_IMPL(ampere_kv, CUDA, module) {
  module.impl("smoke_add", TORCH_FN(smoke_add_cuda));
}

TORCH_LIBRARY_IMPL(ampere_kv, Meta, module) {
  module.impl("smoke_add", TORCH_FN(smoke_add_meta));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  // Python 模块本身不暴露重复的 pybind 函数；导入模块的作用是触发
  // 上面的 torch.library 注册，调用统一走 torch.ops.ampere_kv。
  module.doc() = "AmpereKV v0.1 CUDA Extension 注册模块";
}
