# Weight-Only GEMV API 使用指南

## 发现

TensorRT-LLM **确实有** CUDA Core GEMV 的简洁 API！

位置: `cpp/tensorrt_llm/kernels/weightOnlyBatchedGemv/kernelLauncher.h`

## API 接口

### 核心函数

```cpp
namespace tensorrt_llm::kernels::weight_only {

// 主入口函数
void kernel_launcher(int arch, Params& params, cudaStream_t stream);

// 检查支持
bool is_supported(int arch, KernelType kernel_type);

}
```

### Params 结构

```cpp
struct Params {
    void* act;              // 激活 (FP16/BF16)
    void* act_scale;        // 激活 scale (可选，W4A8 用)
    void* weight;           // 权重 (INT4/INT8, packed)
    void* scales;           // 量化 scales
    void* zeros;            // Zero points (可选)
    void* bias;             // Bias (可选)
    void* out;              // 输出
    float alpha;            // Scaling factor
    int m, n, k;            // 矩阵维度
    int groupsize;          // Group size (groupwise) 或 -1 (per-channel)
    KernelType type;        // Kernel 类型
    bool apply_alpha_in_advance;  // W4A8 用
};
```

### KernelType 枚举

```cpp
enum class KernelType {
    FP16Int8Groupwise,      // FP16 激活 + INT8 权重, groupwise
    BF16Int8Groupwise,      // BF16 激活 + INT8 权重, groupwise
    FP16Int4Groupwise,      // FP16 激活 + INT4 权重, groupwise ← 我们需要这个!
    BF16Int4Groupwise,      // BF16 激活 + INT4 权重, groupwise
    FP16Int8PerChannel,     // FP16 激活 + INT8 权重, per-channel
    BF16Int8PerChannel,     // BF16 激活 + INT8 权重, per-channel
    FP16Int4PerChannel,     // FP16 激活 + INT4 权重, per-channel
    BF16Int4PerChannel      // BF16 激活 + INT4 权重, per-channel
};
```

## 使用示例

### W4A16 Groupwise (FINEGRAINED_SCALE_ONLY)

```cpp
#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/kernelLauncher.h"
#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/common.h"

using namespace tensorrt_llm::kernels::weight_only;

// 准备数据
half* d_A;              // [M, K] FP16 activations
uint8_t* d_B;           // [K, N] INT4 weights (packed)
half* d_scales;         // [N, K/groupsize] FP16 scales
half* d_C;              // [M, N] FP16 output
int M, N, K, groupsize;

// 检查支持
int arch = 90;  // H100/H800
if (!is_supported(arch, KernelType::FP16Int4Groupwise)) {
    fprintf(stderr, "FP16Int4Groupwise not supported on SM%d\n", arch);
    return;
}

// 创建参数
Params params(
    d_A,                              // act
    nullptr,                          // act_scale (不用于 W4A16)
    d_B,                              // weight
    d_scales,                         // scales
    nullptr,                          // zeros (FINEGRAINED_SCALE_ONLY 不需要)
    nullptr,                          // bias (可选)
    d_C,                              // out
    1.0f,                             // alpha
    M, N, K,                          // 矩阵维度
    groupsize,                        // groupsize (例如 128)
    KernelType::FP16Int4Groupwise,    // type
    false                             // apply_alpha_in_advance
);

// 调用 kernel
cudaStream_t stream;
cudaStreamCreate(&stream);

kernel_launcher(arch, params, stream);

cudaStreamSynchronize(stream);
```

### W4A16 PerChannel (PER_COLUMN_SCALE_ONLY)

```cpp
half* d_scales;  // [N] FP16 scales (每列一个)

Params params(
    d_A,
    nullptr,
    d_B,
    d_scales,
    nullptr,
    nullptr,
    d_C,
    1.0f,
    M, N, K,
    -1,                               // groupsize = -1 表示 per-channel
    KernelType::FP16Int4PerChannel,   // 使用 PerChannel 类型
    false
);

kernel_launcher(arch, params, stream);
```

## 架构支持

根据 `kernelLauncher.h`:

| 架构 | SM | 支持的类型 |
|------|-----|-----------|
| Turing | 75-79 | FP16Int4Groupwise, FP16Int4/Int8PerChannel |
| Ampere | 80-89 | 所有 FP16/BF16 + INT4/INT8 组合 |
| **Hopper** | **90-99** | **所有类型 + Hopper 优化 layout** |
| Blackwell | 100+ | 所有类型 + W4A8 支持 |

对于 **SM90 (H100/H800)**:
- 使用 `ColumnMajorInterleavedForHopper` layout
- 支持 `FP16Int4Groupwise` 和 `FP16Int4PerChannel`

## 需要链接的文件

### 源文件 (19 个 .cu)

```bash
cpp/tensorrt_llm/kernels/weightOnlyBatchedGemv/
├── cudaCoreGemm.cu                                           # 核心 GEMM 实现
├── cudaCoreGemmNVFP4.cu                                      # FP4 支持
├── int8SQ.cu                                                 # INT8 smooth quant
├── kernelDispatcherFp16Int4GroupwiseColumnMajorInterleavedTrue.cu
├── kernelDispatcherFp16Int4GroupwiseColumnMajorInterleavedForHopperTrue.cu  # SM90!
├── kernelDispatcherFp16Int4PerChannelColumnMajorInterleavedTrue.cu
├── kernelDispatcherFp16Int4PerChannelColumnMajorInterleavedForHopperTrue.cu # SM90!
├── ... (其他 BF16, INT8 变种)
```

### 头文件

```bash
cpp/tensorrt_llm/kernels/weightOnlyBatchedGemv/
├── kernelLauncher.h     # 主 API
├── common.h             # Params, KernelType
├── kernelDispatcher.h   # 内部调度
├── kernel.h             # Kernel 实现
├── details.h            # 类型细节
├── utility.h            # 工具函数
├── converter.h          # 类型转换
├── cudaCoreGemm.h       # GEMM 接口
├── cudaCoreGemmNVFP4.h  # FP4 接口
└── int8SQ.h             # INT8 SQ
```

### 依赖

最小依赖（已经在我们的构建中）:
- ✅ `tensorrt_llm/common/logger.h`
- ✅ `tensorrt_llm/common/assert.h`
- ✅ `tensorrt_llm/common/cudaUtils.h`
- ✅ CUDA Runtime
- ✅ CUTLASS headers

**新增依赖**（轻量）:
- `tensorrt_llm/common/cudaFp8Utils.h` (只是头文件)
- `cub` (CUDA 自带)

**不需要**:
- ❌ TensorRT runtime
- ❌ NvInfer
- ❌ Complex quantization utils

## 集成到我们的项目

### 步骤 1: 添加源文件到构建

在 `build_w4a16_only.sh` 中:

```cmake
# 添加 GEMV 源文件
file(GLOB GEMV_SOURCES
    "${TRTLLM}/cpp/tensorrt_llm/kernels/weightOnlyBatchedGemv/*.cu"
)

add_executable(w4a16_full_test
    ${TRTLLM}/w4a16_full_test.cu
    ${W4A16_KERNELS}           # CUTLASS kernels
    ${GEMV_SOURCES}            # CUDA Core GEMV kernels
    ${COMMON_SOURCES}
)
```

### 步骤 2: 创建统一的 Runner

创建一个新的测试程序 `w4a16_full_test.cu`，结合两种 kernels:

```cpp
#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm_template.h"
#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/kernelLauncher.h"

void run_w4a16_gemm(
    half* d_A, uint8_t* d_B, half* d_scales, half* d_C,
    int M, int N, int K, int groupsize,
    cudaStream_t stream)
{
    if (M < 64) {
        // 使用 CUDA Core GEMV (小 batch)
        using namespace tensorrt_llm::kernels::weight_only;

        Params params(
            d_A, nullptr, d_B, d_scales, nullptr, nullptr, d_C,
            1.0f, M, N, K, groupsize,
            KernelType::FP16Int4Groupwise,
            false
        );

        kernel_launcher(90, params, stream);

    } else {
        // 使用 CUTLASS (大 batch)
        using namespace tensorrt_llm::kernels::cutlass_kernels;

        CutlassFpAIntBGemmRunner<half, cutlass::uint4b_t,
            cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> runner;

        auto configs = runner.getConfigs();
        auto best_config = selectBestConfig(configs, M, N, K);

        size_t ws_bytes = runner.getWorkspaceSize(M, N, K);
        char* d_ws;
        cudaMalloc(&d_ws, ws_bytes);

        runner.gemm(
            d_A, reinterpret_cast<cutlass::uint4b_t*>(d_B),
            d_scales, nullptr, nullptr, 1.0f,
            d_C, M, N, K, groupsize,
            best_config, d_ws, ws_bytes, stream
        );

        cudaFree(d_ws);
    }
}
```

### 步骤 3: 编译和测试

```bash
./build_w4a16_only.sh

# 测试小 batch (应该用 GEMV)
./build_w4a16_only/w4a16_full_test 16 1024 1024

# 测试大 batch (应该用 CUTLASS)
./build_w4a16_only/w4a16_full_test 512 1024 1024
```

## 预期性能

### GEMV vs CUTLASS

| Batch Size (M) | 使用的 Kernel | 预期性能 |
|----------------|--------------|---------|
| 1-16 | CUDA Core GEMV | 10-50x 比现在快 |
| 17-63 | CUDA Core GEMV | 5-10x 比现在快 |
| 64-127 | CUTLASS (小 tile) | 2-3x 比现在快 |
| 128-511 | CUTLASS (中 tile) | 1.5-2x 比现在快 |
| 512+ | CUTLASS (大 tile) | 1.2-1.5x 比现在快 |

### 切换阈值

建议的阈值:
- **M < 64**: 使用 GEMV
- **M >= 64**: 使用 CUTLASS

可以通过 profiling 找到最优切换点。

## 优势

1. ✅ **API 简洁**: 只需一个函数调用
2. ✅ **依赖轻量**: 不需要 TensorRT runtime
3. ✅ **易于集成**: 19 个 .cu 文件
4. ✅ **已经优化**: TensorRT-LLM 团队维护
5. ✅ **支持完整**: 包括 Hopper 优化

## 下一步

创建 `w4a16_full_test.cu` 实现完整的混合 kernel 方案。

---

**Date**: 2025-11-15
**Status**: API 已找到，准备集成
