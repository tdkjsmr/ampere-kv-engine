#include <torch/extension.h>

// CUDA 实现在 smoke_cuda.cu 中，这里只声明 Python 需要调用的函数。
at::Tensor smoke_add_cuda(const at::Tensor& left, const at::Tensor& right);
// 只读已写好的物理 K/V 与块表；长度包含当前 Token，不负责追加或分配。
at::Tensor paged_decode_cuda(const at::Tensor& query, const at::Tensor& key,
                            const at::Tensor& value, const at::Tensor& table,
                            int64_t length);

// INT8入口只读缓存：每Token、每KV头的K有四组FP16 scale，V有一个。
at::Tensor paged_decode_int8_cuda(const at::Tensor& query, const at::Tensor& key,
                                 const at::Tensor& value, const at::Tensor& key_scale,
                                 const at::Tensor& value_scale, const at::Tensor& table, int64_t length);

// 原地批量量化写入；Decode使用Token数为1的同一入口，不扫描输入数值。
void quantize_write_cuda(const at::Tensor& key, const at::Tensor& value,
                         const at::Tensor& output_key, const at::Tensor& output_value,
                         const at::Tensor& key_scale, const at::Tensor& value_scale,
                         const at::Tensor& table, int64_t start);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("quantize_write", &quantize_write_cuda, "Prefill/Decode INT8量化与分页写入融合");
  // 保留原有 Smoke，便于区分扩展链路错误与 Attention 错误。
  module.def("smoke_add", &smoke_add_cuda, "两个 CUDA Tensor 逐元素相加");
  module.def("paged_decode", &paged_decode_cuda, "单请求 BF16 分页 Decode V0");
  module.def("paged_decode_int8", &paged_decode_int8_cuda, "单请求 INT8 分页 Decode 融合反量化，输出 BF16");
}
