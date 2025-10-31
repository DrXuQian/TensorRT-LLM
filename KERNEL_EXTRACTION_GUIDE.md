# TensorRT-LLM Kernel 提取完整指南

## 目标
从 TensorRT-LLM 中提取特定的 kernel 实现，保持与原始实现 100% 一致，创建最小依赖的独立库。

目标 kernel: `CutlassFpAIntBGemmRunner<half, cutlass::uint4b_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY>`

## Step-by-Step 提取流程

### Step 1: 创建项目结构
```bash
mkdir extracted_fp16_int4_gemm
cd extracted_fp16_int4_gemm
mkdir -p include/tensorrt_llm/common
mkdir -p include/cutlass_extensions
mkdir -p src
mkdir -p test
mkdir -p build
```

### Step 2: 复制核心 Kernel 文件

#### 2.1 复制主要 kernel 头文件
```bash
# 从 TensorRT-LLM 源码复制
cp /path/to/tensorrt-llm/cpp/tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h include/
cp /path/to/tensorrt-llm/cpp/tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm_template.h include/
```

#### 2.2 复制 cutlass_extensions 目录
```bash
# 完整复制 cutlass_extensions（包含必要的 CUTLASS 扩展）
cp -r /path/to/tensorrt-llm/cpp/tensorrt_llm/kernels/cutlass_kernels/cutlass_extensions include/
```

#### 2.3 复制必要的 common 头文件
```bash
cp /path/to/tensorrt-llm/cpp/include/tensorrt_llm/common/*.h include/tensorrt_llm/common/
```

### Step 3: 创建 Kernel 实例化文件

创建 `src/fp16_int4_gemm_kernel.cu`:
```cpp
#include "../include/fpA_intB_gemm.h"
#include "../include/fpA_intB_gemm_template.h"

// 实例化需要的具体 kernel
template class tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<
    half, cutlass::uint4b_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY>;
```

### Step 4: 创建 CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.18)
project(extracted_fp16_int4_gemm CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# 关键：排除 SM90 以避免未定义符号
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89" CACHE STRING "CUDA architectures")

# 设置 CUTLASS 路径
set(CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/cutlass" CACHE PATH "Path to CUTLASS")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cutlass_extensions/include
    ${CUTLASS_DIR}/include
    ${CUTLASS_DIR}/tools/util/include  # for packed_stride.hpp
    ${CUDAToolkit_INCLUDE_DIRS}
)

# 编译标志
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math")

# 定义
add_definitions(-DENABLE_FP8=1)
add_definitions(-DENABLE_BF16=1)
```

## 编译过程中遇到的问题及解决方案

### 问题 1: 缺少 common.h
**错误**: `fatal error: tensorrt_llm/common/common.h: No such file or directory`

**解决方案**:
```bash
cp /path/to/tensorrt-llm/cpp/include/tensorrt_llm/common/common.h include/tensorrt_llm/common/
```

### 问题 2: cutlass_extensions 包含路径错误
**错误**: `fatal error: tensorrt_llm/cutlass_extensions/include/xxx.h: No such file or directory`

**解决方案**: 修改 CMakeLists.txt，添加正确的包含路径：
```cmake
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cutlass_extensions/include  # 添加这行
)
```

### 问题 3: 缺少 NvInferRuntime.h
**错误**: `fatal error: NvInferRuntime.h: No such file or directory`

**解决方案**: 创建存根文件 `include/NvInferRuntime.h`:
```cpp
#pragma once
namespace nvinfer1 {
    enum class DataType : int32_t {
        kFLOAT = 0,
        kHALF = 1,
        kINT8 = 2,
        kINT32 = 3,
        kBOOL = 4,
        kUINT8 = 5,
        kFP8 = 6,
        kBF16 = 7,
        kINT64 = 8,
        kINT4 = 9,
    };
}
```

### 问题 4: 缺少 packed_stride.hpp
**错误**: `fatal error: cute/stride.hpp: No such file or directory`

**解决方案**: 添加 CUTLASS tools 路径到 CMakeLists.txt:
```cmake
include_directories(
    ${CUTLASS_DIR}/tools/util/include  # 添加这行
)
```

### 问题 5: SM90 未定义符号
**错误**: 链接时出现大量 SM90 相关的未定义符号

**解决方案 1**: 在 CMakeLists.txt 中排除 SM90：
```cmake
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89")  # 不包含 90
```

**解决方案 2**: 修改 `fpA_intB_gemm_template.h`，在 SM90 分支抛出异常：
```cpp
else if (sm_ == 90)
{
    throw std::runtime_error(
        "[TensorRT LLM Error][CutlassFpAIntBGemmRunner][dispatch_to_arch] "
        "SM90 not supported in this extraction. Please use SM89 or below.");
}
```

**解决方案 3**: 注释掉 SM90 模板包含（如果存在）：
```cpp
// #include "../fpA_intB_gemm_template_sm90.h"
```

### 问题 6: Logger 相关符号未定义
**错误**:
```
undefined reference to `tensorrt_llm::common::Logger::getLogger()'
undefined reference to `tensorrt_llm::common::fmtstr_(...)'
```

**解决方案**: 创建 `src/logger_stub.cpp`:
```cpp
#include "tensorrt_llm/common/logger.h"
namespace tensorrt_llm {
namespace common {
    Logger::Logger() {}
    Logger* Logger::getLogger() {
        static Logger instance;
        return &instance;
    }
    void fmtstr_(const char* format, fmtstr_allocator alloc, void* target, va_list args) {
        // Empty stub
    }
}
}
```

### 问题 7: get_candidate_configs 未定义
**错误**: `undefined reference to get_candidate_configs`

**解决方案**: 创建 `src/missing_implementations.cpp`:
```cpp
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.h"

namespace tensorrt_llm {
namespace kernels {
namespace cutlass_kernels {

std::vector<tensorrt_llm::cutlass_extensions::CutlassGemmConfig>
get_candidate_configs(int sm, int split_k_limit,
    tensorrt_llm::cutlass_extensions::CutlassGemmConfig::CandidateConfigTypeParam config_type)
{
    std::vector<tensorrt_llm::cutlass_extensions::CutlassGemmConfig> configs;
    tensorrt_llm::cutlass_extensions::CutlassGemmConfig config;
    config.tile_config_sm80 = tensorrt_llm::cutlass_extensions::CutlassTileConfig::ChooseWithHeuristic;
    config.split_k_style = tensorrt_llm::cutlass_extensions::SplitKStyle::NO_SPLIT_K;
    config.split_k_factor = 1;
    config.stages = 3;
    configs.push_back(config);
    return configs;
}

}}}
```

### 问题 8: SM120 GPU 无法运行（向前兼容问题）
**错误**: Kernel 在 SM120 GPU 上执行失败

**根本原因**: getSMVersion() 返回 120，但调度代码将 sm_ >= 100 路由到 SM80，需要让 SM120 使用 SM89 代码

**解决方案**: 修改 `fpA_intB_gemm_template.h` 的调度逻辑：
```cpp
// 原始代码
else if ((sm_ >= 80 && sm_ < 89) || sm_ >= 100)
{
    // SM80 path
}
else if (sm_ == 89)
{
    // SM89 path
}

// 修改为
else if ((sm_ >= 80 && sm_ < 89))
{
    // SM80 path
}
else if (sm_ == 89 || sm_ >= 100)  // SM100+ 使用 SM89 代码（向前兼容）
{
    // SM89 path
}
```

## 完整的 CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.18)
project(extracted_fp16_int4_gemm CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

set(CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/cutlass" CACHE PATH "Path to CUTLASS")
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89" CACHE STRING "CUDA architectures")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cutlass_extensions/include
    ${CUTLASS_DIR}/include
    ${CUTLASS_DIR}/tools/util/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wno-deprecated-declarations")

add_definitions(-DTLLM_LOG_LEVEL=TLLM_LOG_LEVEL_ERROR)
add_definitions(-DENABLE_FP8=1)
add_definitions(-DENABLE_BF16=1)

add_library(tensorrt_llm_fp16_int4_gemm SHARED
    src/fp16_int4_gemm_kernel.cu
    src/gemm_wrapper.cpp
    src/missing_implementations.cpp
    src/logger_stub.cpp
)

target_link_libraries(tensorrt_llm_fp16_int4_gemm
    CUDA::cudart
    CUDA::cuda_driver
)
```

## 测试验证

### 创建测试程序 test/test_kernel.cu:
```cpp
#include "../include/fpA_intB_gemm.h"
using namespace tensorrt_llm::kernels::cutlass_kernels;

int main() {
    // 创建 runner（这是原始的 TensorRT-LLM kernel）
    CutlassFpAIntBGemmRunner<half, cutlass::uint4b_t,
        cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> runner;

    // 设置配置
    tensorrt_llm::cutlass_extensions::CutlassGemmConfig config;
    config.tile_config_sm80 = CutlassTileConfig::CtaShape128x128x64_WarpShape128x32x64;
    config.split_k_style = SplitKStyle::NO_SPLIT_K;

    // 运行 kernel
    runner.gemm(...);
}
```

## 编译和运行

```bash
cd build
cmake ..
make -j

# 运行测试
./test_kernel
```

## 关键要点总结

1. **保持代码完整性**: 直接复制 TensorRT-LLM 的源文件，不要尝试重写
2. **处理架构兼容性**: 排除 SM90 以避免未定义符号，使用 SM89 代码支持更新的 GPU
3. **创建必要的存根**: 为缺失的依赖创建最小实现
4. **正确设置包含路径**: 确保所有头文件路径正确
5. **修改调度逻辑**: 确保新架构（如 SM120）能正确路由到兼容的代码路径

## 适配到其他 TensorRT-LLM 版本

1. **识别 kernel 位置**: 在自定义版本中找到对应的 kernel 文件
2. **检查依赖变化**: 不同版本可能有不同的依赖结构
3. **调整架构支持**: 根据目标 GPU 调整 CMAKE_CUDA_ARCHITECTURES
4. **处理版本特定问题**: 不同版本可能有特定的编译问题，按照上述模式解决

这个方法可以提取任何 TensorRT-LLM kernel 并创建独立的最小依赖库。