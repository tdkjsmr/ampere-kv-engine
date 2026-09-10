#include <ATen/ATen.h>
#include <c10/util/BFloat16.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>

namespace {
constexpr int kDim = 128;
constexpr int kBlockSize = 16;
constexpr int kSegmentSize = 64;  // 与此前 256 段长做单变量实验，不代表最终最优值。

// 每个线程块负责一个 Query 头的一段历史，每个线程负责该头的一个维度。
// 不创建连续历史副本，不展开 GQA 的 K/V，不使用 Tensor Core 或异步搬运。
__global__ void paged_decode_kernel(
    const c10::BFloat16* query, const c10::BFloat16* key,
    const c10::BFloat16* value, const int64_t* table,
    c10::BFloat16* output, float* partials, int64_t length,
    int kv_heads, int group_size, int segments) {
  const int dim = threadIdx.x;
  const int lane = dim % 32;
  const int warp = dim / 32;
  const int query_head = blockIdx.x;
  const int kv_head = query_head / group_size;
  const float q = static_cast<float>(query[query_head * kDim + dim]);
  float accumulator = 0.0f;  // 当前维度的加权 V 分子，始终保留 FP32。
  // 固定 128 线程，即 4 个完整 warp；共享内存只保存各 warp 的部分和。
  __shared__ float warp_sums[kDim / 32];
  __shared__ float maximum, denominator, old_scale, new_weight;
  if (dim == 0) {
    maximum = -INFINITY;
    denominator = 0.0f;
  }
  __syncthreads();

  const int64_t begin = static_cast<int64_t>(blockIdx.y) * kSegmentSize;
  const int64_t end = length < begin + kSegmentSize ? length : begin + kSegmentSize;
  for (int64_t token = begin; token < end; ++token) {
    // 布局 [物理块, KV 头, 块内 Token, 维度]；索引使用 64 位避免乘法溢出。
    const int64_t physical = table[token / kBlockSize];
    const int offset = token % kBlockSize;
    const int64_t index = ((physical * kv_heads + kv_head) * kBlockSize + offset) * kDim + dim;
    float sum = q * static_cast<float>(key[index]);
    // 所有 lane 都参与洗牌，不在 lane == 0 分支中调用全掩码 shuffle。
    // warp 内通过寄存器交换归约，只有 lane 0 的最终和用于跨 warp 合并。
    for (int delta = 16; delta > 0; delta /= 2) {
      sum += __shfl_down_sync(0xffffffffu, sum, delta);
    }
    if (lane == 0) warp_sums[warp] = sum;
    // shuffle 只同步同一 warp；读其他 warp 的共享部分和前仍须块级屏障。
    __syncthreads();
    if (warp == 0) {
      // 第一个完整 warp 合并 4 个部分和，其余 lane 补零；全掩码仍然有效。
      sum = lane < kDim / 32 ? warp_sums[lane] : 0.0f;
      for (int delta = 16; delta > 0; delta /= 2) {
        sum += __shfl_down_sync(0xffffffffu, sum, delta);
      }
    }
    if (dim == 0) {
      // 浮点加法次序与原树形归约不同，必须重新跑数值与模型对照。
      const float score = sum * rsqrtf(static_cast<float>(kDim));
      // 在线 softmax：最大分数变化时，旧分子和分母都乘同一个缩放系数。
      // 无需保存所有分数；减去最大值避免直接 exp(score) 溢出。
      const float next_maximum = fmaxf(maximum, score);
      old_scale = expf(maximum - next_maximum);
      new_weight = expf(score - next_maximum);
      denominator = denominator * old_scale + new_weight;
      maximum = next_maximum;
    }
    __syncthreads();
    accumulator = accumulator * old_scale + new_weight * static_cast<float>(value[index]);
    // 保证所有线程读完本轮系数，下一轮线程 0 才能覆盖它们。
    __syncthreads();
  }
  if (partials == nullptr) {
    // 单段保留原路径：不分配临时张量，也不启动合并内核。
    output[query_head * kDim + dim] = c10::BFloat16(accumulator / denominator);
  } else {
    // 布局 [Query头, 段, 128维分子 + 最大值 + 分母]，全程保存 FP32。
    // 不能先转 BF16，也不能只保存每段归一化后的输出再取平均。
    const int64_t base = (static_cast<int64_t>(query_head) * segments + blockIdx.y) * (kDim + 2);
    partials[base + dim] = accumulator;
    if (dim == 0) {
      partials[base + kDim] = maximum;
      partials[base + kDim + 1] = denominator;
    }
  }
}

// 同一 CUDA 流上的第二次启动保证所有局部结果已写好，不需要 CPU 同步。
__global__ void merge_decode_kernel(const float* partials, c10::BFloat16* output, int segments) {
  const int dim = threadIdx.x;
  const int64_t base = static_cast<int64_t>(blockIdx.x) * segments * (kDim + 2);
  float maximum = -INFINITY;
  for (int segment = 0; segment < segments; ++segment) {
    maximum = fmaxf(maximum, partials[base + static_cast<int64_t>(segment) * (kDim + 2) + kDim]);
  }
  float numerator = 0.0f, denominator = 0.0f;
  for (int segment = 0; segment < segments; ++segment) {
    const int64_t offset = base + static_cast<int64_t>(segment) * (kDim + 2);
    // 各段使用各自最大值计算过指数；重新缩放到共同最大值后才能相加。
    const float scale = expf(partials[offset + kDim] - maximum);
    numerator += scale * partials[offset + dim];
    denominator += scale * partials[offset + kDim + 1];
  }
  output[blockIdx.x * kDim + dim] = c10::BFloat16(numerator / denominator);
}
}  // 匿名命名空间：内部实现不暴露给其他编译单元。

at::Tensor paged_decode_cuda(const at::Tensor& query, const at::Tensor& key,
                            const at::Tensor& value, const at::Tensor& table,
                            int64_t length) {
  TORCH_CHECK(query.is_cuda() && key.is_cuda() && value.is_cuda() && table.is_cuda(), "输入必须全部位于 CUDA");
  TORCH_CHECK(query.device() == key.device() && key.device() == value.device() && key.device() == table.device(), "输入必须位于同一设备");
  TORCH_CHECK(query.scalar_type() == at::kBFloat16 && key.scalar_type() == at::kBFloat16 && value.scalar_type() == at::kBFloat16, "Q/K/V 必须是 BF16");
  TORCH_CHECK(table.scalar_type() == at::kLong && table.dim() == 1, "块表必须是一维 int64");
  TORCH_CHECK(query.is_contiguous() && key.is_contiguous() && value.is_contiguous() && table.is_contiguous(), "V0 只接受连续布局张量");
  TORCH_CHECK(query.dim() == 4 && query.size(0) == 1 && query.size(2) == 1 && query.size(3) == kDim, "Q 必须是 [1, Query头数, 1, 128]");
  TORCH_CHECK(key.dim() == 4 && key.sizes() == value.sizes() && key.size(0) > 0 && key.size(2) == kBlockSize && key.size(3) == kDim, "K/V 必须是 [物理块数, KV头数, 16, 128]");
  const int64_t q_heads = query.size(1), kv_heads = key.size(1);
  TORCH_CHECK(q_heads > 0 && q_heads <= 1024 && kv_heads > 0 && q_heads % kv_heads == 0, "V0 要求 Query 头数不超过 1024 且是 KV 头数的正整数倍");
  TORCH_CHECK(length > 0 && (length - 1) / kBlockSize < table.numel(), "有效长度必须为正且块表必须足够长");
  const int64_t segment_count = (length - 1) / kSegmentSize + 1;
  TORCH_CHECK(segment_count <= 65535, "分段数超过二维 CUDA 网格上限");
  const int segments = static_cast<int>(segment_count);
  const c10::cuda::CUDAGuard guard(query.device());
  // V0 为安全先检查有效块号。这两次 item 会同步，不支持 CUDA Graph 捕获。
  // 只检查使用到的块表前缀，不读取未使用的尾部；块归属仍由调用方保证。
  const auto used_table = table.narrow(0, 0, (length - 1) / kBlockSize + 1);
  TORCH_CHECK(used_table.min().item<int64_t>() >= 0 && used_table.max().item<int64_t>() < key.size(0), "物理块编号越界");
  auto output = at::empty_like(query);
  const auto stream = c10::cuda::getCurrentCUDAStream(query.get_device()).stream();
  // 长历史才建立局部结果；临时存储分配和额外启动均计入现有调用基线。
  at::Tensor partials;
  float* partial_ptr = nullptr;
  if (segments > 1) {
    partials = at::empty({q_heads, segment_count, kDim + 2}, query.options().dtype(at::kFloat));
    partial_ptr = partials.data_ptr<float>();
  }
  const dim3 grid(static_cast<unsigned int>(q_heads), static_cast<unsigned int>(segments));
  paged_decode_kernel<<<grid, kDim, 0, stream>>>(
      query.data_ptr<c10::BFloat16>(), key.data_ptr<c10::BFloat16>(),
      value.data_ptr<c10::BFloat16>(), table.data_ptr<int64_t>(),
      output.data_ptr<c10::BFloat16>(), partial_ptr, length, static_cast<int>(kv_heads),
      static_cast<int>(q_heads / kv_heads), segments);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (segments > 1) {
    merge_decode_kernel<<<static_cast<unsigned int>(q_heads), kDim, 0, stream>>>(
        partial_ptr, output.data_ptr<c10::BFloat16>(), segments);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}
