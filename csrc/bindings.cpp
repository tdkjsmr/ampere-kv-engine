#include <torch/extension.h>

// CUDA 实现在 smoke_cuda.cu 中，这里只声明 Python 需要调用的函数。
at::Tensor smoke_add_cuda(const at::Tensor& left, const at::Tensor& right);
// 只读已写好的物理 K/V 与块表；长度包含当前 Token，不负责追加或分配。
// v3=true 时改用"一个线程块服务同一 GQA 组四个 Query 头"的实验内核，默认 false 仍是 V1。
at::Tensor paged_decode_cuda(const at::Tensor& query, const at::Tensor& key,
                            const at::Tensor& value, const at::Tensor& table,
                            int64_t length, bool v3);

// INT8入口只读缓存：每Token、每KV头的K有四组FP16 scale，V有一个。
at::Tensor paged_decode_int8_cuda(const at::Tensor& query, const at::Tensor& key,
                                 const at::Tensor& value, const at::Tensor& key_scale,
                                 const at::Tensor& value_scale, const at::Tensor& table, int64_t length, bool v3);

// 模型内部入口不做宿主标量取回；块号由内核在访存前保护，非法块表会异步失败并终止运行。
at::Tensor paged_decode_internal_cuda(const at::Tensor& query, const at::Tensor& key,
                                     const at::Tensor& value, const at::Tensor& table, int64_t length, bool v3);

at::Tensor paged_decode_int8_internal_cuda(const at::Tensor& query, const at::Tensor& key,
                                          const at::Tensor& value, const at::Tensor& key_scale,
                                          const at::Tensor& value_scale, const at::Tensor& table, int64_t length, bool v3);

// 原地批量量化写入；Decode使用Token数为1的同一入口，不扫描输入数值。
void quantize_write_cuda(const at::Tensor& key, const at::Tensor& value,
                         const at::Tensor& output_key, const at::Tensor& output_value,
                         const at::Tensor& key_scale, const at::Tensor& value_scale,
                         const at::Tensor& table, int64_t start);

// BF16原地批量写入；布局与分工同量化写入，只搬运不改变数值。
void bf16_write_cuda(const at::Tensor& key, const at::Tensor& value,
                     const at::Tensor& output_key, const at::Tensor& output_value,
                     const at::Tensor& table, int64_t start);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("quantize_write", &quantize_write_cuda, "Prefill/Decode INT8量化与分页写入融合");
  module.def("bf16_write", &bf16_write_cuda, "Prefill/Decode BF16分页写入融合");
  // 保留原有 Smoke，便于区分扩展链路错误与 Attention 错误。
  module.def("smoke_add", &smoke_add_cuda, "两个 CUDA Tensor 逐元素相加");
  module.def("paged_decode", &paged_decode_cuda, "单请求 BF16 分页 Decode V0", py::arg("v3") = false);
  module.def("paged_decode_int8", &paged_decode_int8_cuda, "单请求 INT8 分页 Decode 融合反量化，输出 BF16", py::arg("v3") = false);
  // 内部入口只给模型生成路径使用；独立算子检查与外部调用请继续用上面两个受检查入口。
  // v3 需要显式传 True，且只支持 4 倍 GQA；默认 False 仍是 V1，未测量前不替换默认路径。
  module.def("paged_decode_internal", &paged_decode_internal_cuda, "模型内部 BF16 分页 Decode；无宿主标量取回，块号由内核在访存前保护", py::arg("v3") = false);
  module.def("paged_decode_int8_internal", &paged_decode_int8_internal_cuda, "模型内部 INT8 分页 Decode；无宿主标量取回，块号由内核在访存前保护", py::arg("v3") = false);
}
