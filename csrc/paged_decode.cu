#include <ATen/ATen.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <type_traits>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>

namespace {
constexpr int kDim = 128;
constexpr int kBlockSize = 16;
constexpr int kSegmentSize = 64;  // 与此前 256 段长做单变量实验，不代表最终最优值。

// 一个线程块的四个 warp 分别处理段内不同 Token；每线程仍负责四维。
// 不创建连续历史副本，不展开 GQA 的 K/V，不使用 Tensor Core 或异步搬运。
// 只实例化 BF16 与 INT8 两种读取方式，归约和分段合并共用；不是通用类型派发框架。
template <typename KV>
__global__ void paged_decode_kernel(
    const c10::BFloat16* query, const KV* key,
    const KV* value, const int64_t* table,
    const c10::Half* key_scale, const c10::Half* value_scale,
    c10::BFloat16* output, float* partials, int64_t length,
    int kv_heads, int group_size, int segments) {
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;  // 固定启动四个完整 warp，shuffle 只在各自 warp 内执行。
  // 只暂存局部结果，不搬运 K/V：四份 FP32 分子、最大值和分母，共 2080 字节。
  __shared__ float local[4][kDim + 2];
  const int query_head = blockIdx.x;
  const int kv_head = query_head / group_size;
  // 固定四元素数组配合展开供编译器标量化；实际寄存器占用以编译结果为准。
  float q[4];
  float accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    q[i] = static_cast<float>(query[query_head * kDim + lane + i * 32]);
  }
  // 仅 lane 0 更新段内最大值与分母，其他 lane 通过 shuffle 获取所需数据。
  float maximum = -INFINITY, denominator = 0.0f;

  const int64_t begin = static_cast<int64_t>(blockIdx.y) * kSegmentSize;
  const int64_t end = length < begin + kSegmentSize ? length : begin + kSegmentSize;
  // 交错分工：warp 0 处理 0、4、8……，warp 1 处理 1、5、9……。
  // 完整 64 Token 段中，每个 warp 只串行处理 16 个 Token。
  for (int64_t token = begin + warp; token < end; token += 4) {
    // 布局 [物理块, KV 头, 块内 Token, 维度]；索引使用 64 位避免乘法溢出。
    const int64_t physical = table[token / kBlockSize];
    const int offset = token % kBlockSize;
    const int64_t index = ((physical * kv_heads + kv_head) * kBlockSize + offset) * kDim + lane;
    float k_scale = 1.0f, v_scale = 1.0f;
    if constexpr (std::is_same_v<KV, int8_t>) {
      // K scale末维4、V末维1；必须使用KV头而非Query头。
      const int64_t scale_index = (physical * kv_heads + kv_head) * kBlockSize + offset;
      // lane 0..3各读一组K scale，下面按维度组编号广播，不需要共享内存。
      if (lane < 4) k_scale = static_cast<float>(key_scale[scale_index * 4 + lane]);
      if (lane == 0) {
        v_scale = static_cast<float>(value_scale[scale_index]);
      }
      v_scale = __shfl_sync(0xffffffffu, v_scale, 0);
    }
    float sum = 0.0f;
    // 每轮 i 中相邻 lane 读取相邻维度：lane、lane+32、lane+64、lane+96。
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float k = static_cast<float>(key[index + i * 32]);
      // lane+i*32属于第i组；全部lane参与shuffle，从lane i获取该组scale。
      if constexpr (std::is_same_v<KV, int8_t>) k *= __shfl_sync(0xffffffffu, k_scale, i);
      sum += q[i] * k;
    }
    // 先合并线程内四项，再做 warp 归约；只使用 lane 0 的最终和。
    for (int delta = 16; delta > 0; delta /= 2) {
      sum += __shfl_down_sync(0xffffffffu, sum, delta);
    }
    float old_scale = 0.0f, new_weight = 0.0f;
    if (lane == 0) {
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
    // 分支外由全部 lane 广播系数，不依赖隐式锁步或共享内存。
    old_scale = __shfl_sync(0xffffffffu, old_scale, 0);
    new_weight = __shfl_sync(0xffffffffu, new_weight, 0);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float v = static_cast<float>(value[index + i * 32]);
      if constexpr (std::is_same_v<KV, int8_t>) v *= v_scale;
      accumulator[i] = accumulator[i] * old_scale + new_weight * v;
    }
    // Token 循环内仍只有私有状态与 warp shuffle，没有块级屏障。
  }
  // 即使没有分到 Token，也写出初始值：分子和分母为零，最大值为负无穷。
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    local[warp][lane + i * 32] = accumulator[i];
  }
  if (lane == 0) {
    local[warp][kDim] = maximum;
    local[warp][kDim + 1] = denominator;
  }
  // 所有线程都必须到达这里；保证四份结果可见后，只有 warp 0 负责合并。
  __syncthreads();
  if (warp != 0) return;

  maximum = -INFINITY;
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    maximum = fmaxf(maximum, local[w][kDim]);
  }
  denominator = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) accumulator[i] = 0.0f;
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    // 空 warp 不参与指数计算；尤其长度为 1 或末段不足四个 Token 时不可遗漏。
    if (local[w][kDim + 1] > 0.0f) {
      const float scale = expf(local[w][kDim] - maximum);
      denominator += scale * local[w][kDim + 1];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        accumulator[i] += scale * local[w][lane + i * 32];
      }
    }
  }
  if (partials == nullptr) {
    // 单段保留原路径：不分配临时张量，也不启动合并内核。
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      output[query_head * kDim + lane + i * 32] = c10::BFloat16(accumulator[i] / denominator);
    }
  } else {
    // 布局 [Query头, 段, 128维分子 + 最大值 + 分母]，全程保存 FP32。
    // 不能先转 BF16，也不能只保存每段归一化后的输出再取平均。
    const int64_t base = (static_cast<int64_t>(query_head) * segments + blockIdx.y) * (kDim + 2);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      partials[base + lane + i * 32] = accumulator[i];
    }
    if (lane == 0) {
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

// 共用形状检查、块号检查、工作区与启动；两个公开入口固定各自精度。
static at::Tensor paged_decode_impl(const at::Tensor& query, const at::Tensor& key,
                            const at::Tensor& value, const at::Tensor& table,
                            int64_t length, bool quantized,
                            const at::Tensor& key_scale, const at::Tensor& value_scale) {
  TORCH_CHECK(query.is_cuda() && key.is_cuda() && value.is_cuda() && table.is_cuda(), "输入必须全部位于 CUDA");
  TORCH_CHECK(query.device() == key.device() && key.device() == value.device() && key.device() == table.device(), "输入必须位于同一设备");
  const auto kv_type = quantized ? at::kChar : at::kBFloat16;
  TORCH_CHECK(query.scalar_type() == at::kBFloat16 && key.scalar_type() == kv_type && value.scalar_type() == kv_type, "Q 必须是 BF16，K/V 类型必须匹配所选入口");
  TORCH_CHECK(table.scalar_type() == at::kLong && table.dim() == 1, "块表必须是一维 int64");
  TORCH_CHECK(query.is_contiguous() && key.is_contiguous() && value.is_contiguous() && table.is_contiguous(), "V0 只接受连续布局张量");
  TORCH_CHECK(query.dim() == 4 && query.size(0) == 1 && query.size(2) == 1 && query.size(3) == kDim, "Q 必须是 [1, Query头数, 1, 128]");
  TORCH_CHECK(key.dim() == 4 && key.sizes() == value.sizes() && key.size(0) > 0 && key.size(2) == kBlockSize && key.size(3) == kDim, "K/V 必须是 [物理块数, KV头数, 16, 128]");
  const int64_t q_heads = query.size(1), kv_heads = key.size(1);
  TORCH_CHECK(q_heads > 0 && q_heads <= 1024 && kv_heads > 0 && q_heads % kv_heads == 0, "V0 要求 Query 头数不超过 1024 且是 KV 头数的正整数倍");
  TORCH_CHECK(length > 0 && (length - 1) / kBlockSize < table.numel(), "有效长度必须为正且块表必须足够长");
  if (quantized) {
    TORCH_CHECK(key_scale.is_cuda() && value_scale.is_cuda() && key_scale.device() == query.device() && value_scale.device() == query.device(), "scale 必须与 Q 同一 CUDA 设备");
    TORCH_CHECK(key_scale.scalar_type() == at::kHalf && value_scale.scalar_type() == at::kHalf && key_scale.is_contiguous() && value_scale.is_contiguous(), "scale 必须是连续 FP16 张量");
    TORCH_CHECK(key_scale.dim() == 4 && key_scale.size(0) == key.size(0) && key_scale.size(1) == kv_heads && key_scale.size(2) == kBlockSize && key_scale.size(3) == 4, "K scale必须是 [物理块数, KV头数, 16, 4]，请使用分组K存储");
    TORCH_CHECK(value_scale.dim() == 4 && value_scale.size(0) == key.size(0) && value_scale.size(1) == kv_heads && value_scale.size(2) == kBlockSize && value_scale.size(3) == 1, "V scale必须是 [物理块数, KV头数, 16, 1]");
    // 有效槽位的整数范围与有限正 scale 由已验证的量化写入保证；此处不扫描数值。
    // 未写尾部允许哨兵值，内核只按有效长度读取；此入口不是不可信数据清洗器。
  }
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
  // 四个 warp 独立计算，段末才合并；跨段合并内核保持原样。
  if (quantized) {
    paged_decode_kernel<int8_t><<<grid, 128, 0, stream>>>(
        query.data_ptr<c10::BFloat16>(), key.data_ptr<int8_t>(), value.data_ptr<int8_t>(),
        table.data_ptr<int64_t>(), key_scale.data_ptr<c10::Half>(), value_scale.data_ptr<c10::Half>(),
        output.data_ptr<c10::BFloat16>(), partial_ptr, length, static_cast<int>(kv_heads),
        static_cast<int>(q_heads / kv_heads), segments);
  } else {
    paged_decode_kernel<c10::BFloat16><<<grid, 128, 0, stream>>>(
        query.data_ptr<c10::BFloat16>(), key.data_ptr<c10::BFloat16>(),
        value.data_ptr<c10::BFloat16>(), table.data_ptr<int64_t>(), nullptr, nullptr,
        output.data_ptr<c10::BFloat16>(), partial_ptr, length, static_cast<int>(kv_heads),
        static_cast<int>(q_heads / kv_heads), segments);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (segments > 1) {
    merge_decode_kernel<<<static_cast<unsigned int>(q_heads), kDim, 0, stream>>>(
        partial_ptr, output.data_ptr<c10::BFloat16>(), segments);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}

at::Tensor paged_decode_cuda(const at::Tensor& query, const at::Tensor& key,
                            const at::Tensor& value, const at::Tensor& table, int64_t length) {
  return paged_decode_impl(query, key, value, table, length, false, at::Tensor(), at::Tensor());
}

at::Tensor paged_decode_int8_cuda(const at::Tensor& query, const at::Tensor& key,
                                 const at::Tensor& value, const at::Tensor& key_scale,
                                 const at::Tensor& value_scale, const at::Tensor& table, int64_t length) {
  return paged_decode_impl(query, key, value, table, length, true, key_scale, value_scale);
}
