cmake_minimum_required(VERSION 3.24)

project(
  ampere_kv_engine
  VERSION 0.1.0
  LANGUAGES CXX CUDA
)

# v0.1 固定使用 C++17；CUDA 架构固定为 RTX 3090/A10 对应的 sm_86。
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

# CMake 路径主要服务于云端环境诊断和 IDE 索引。
# 正式 Python Extension 默认通过 setup.py / torch.utils.cpp_extension 构建。
find_package(Torch REQUIRED)

add_library(
  ampere_kv_cuda_smoke
  SHARED
  csrc/bindings.cpp
  csrc/smoke_cuda.cu
)

target_include_directories(
  ampere_kv_cuda_smoke
  PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc"
)

target_link_libraries(
  ampere_kv_cuda_smoke
  PRIVATE
  "${TORCH_LIBRARIES}"
)

# bindings.cpp 使用 PyTorch Extension 约定的模块名宏。
# setup.py 会自动注入该宏，CMake 诊断路径需要显式提供。
target_compile_definitions(
  ampere_kv_cuda_smoke
  PRIVATE
  TORCH_EXTENSION_NAME=ampere_kv_cuda_smoke
)

target_compile_options(
  ampere_kv_cuda_smoke
  PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --lineinfo>
)

# torch.ops.load_library 直接加载共享库，因此不需要 CMake 默认的 lib 前缀。
set_target_properties(
  ampere_kv_cuda_smoke
  PROPERTIES
  PREFIX ""
)
