# W4A16 Performance 问题详细分析

## 问题

提取的 W4A16 SM90 kernels **性能很差**，即使尝试了所有 GemmConfig 也无法达到预期性能。

## 根本原因

经过详细分析，发现了 **3 个关键问题**：

---

## 问题 1: TensorRT-LLM 使用 **两套** Kernel 实现

### 发现

TensorRT-LLM 对 W4A16 有 **两种完全不同的 kernel 实现**：

#### 1.1 CUTLASS Kernels (我们提取的)
- 路径: `generated_kernels_w4a16_only/*.cu`
- 类型: CUTLASS 3.x 基于 TMA/WGMMA 的 Hopper kernels
- 特点:
  - 使用 SM90 专有指令 (TMA, WGMMA)
  - 针对大矩阵优化
  - 需要 Cluster scheduling

#### 1.2 CUDA Core Kernels (我们**缺失**的!)
- 路径: `cpp/tensorrt_llm/kernels/weightOnlyBatchedGemv/`
- 类型: 手写的 CUDA Core kernels
- 特点:
  - **针对小 batch (M=1-64) 优化**
  - 使用 CUDA Core 而非 Tensor Core
  - GEMV (矩阵-向量乘法) 优化
  - 延迟更低

### 证据

查看 `cpp/tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.cpp:357-365`:

```cpp
// add cuda kernel profiler to tactics for weight-only plugins
if (config & CutlassGemmConfig::WEIGHT_ONLY)
{
    if (tiles.size() > 0 && !(config & CutlassGemmConfig::GROUPED_GEMM))
    {
        CutlassGemmConfig CudaKernelConfig(
            tiles[0], MainloopScheduleType::AUTO, EpilogueScheduleType::AUTO, ClusterShape::ClusterShape_1x1x1);
        CudaKernelConfig.enableCudaKernel = true;  // ← 启用 CUDA kernel!
        candidate_configs.push_back(CudaKernelConfig);
    }
}
```

这段代码说明：
1. TensorRT-LLM 会为 weight-only (包括 W4A16) 添加一个 **CUDA kernel 配置**
2. 这个配置会在 runner 的候选列表中
3. TensorRT-LLM 会根据矩阵尺寸自动选择 CUTLASS 或 CUDA kernel

### CUDA Kernel 文件列表

```
cpp/tensorrt_llm/kernels/weightOnlyBatchedGemv/
├── cudaCoreGemm.cu                                          # 核心 GEMV 实现
├── cudaCoreGemmNVFP4.cu                                     # FP4 变种
├── kernelDispatcherFp16Int4GroupwiseColumnMajorInterleavedTrue.cu       # FP16+INT4
├── kernelDispatcherFp16Int4GroupwiseColumnMajorInterleavedForHopperTrue.cu  # FP16+INT4 Hopper
├── kernelDispatcherFp16Int4PerChannelColumnMajorInterleavedTrue.cu
├── kernelDispatcherFp16Int4PerChannelColumnMajorInterleavedForHopperTrue.cu
└── ... (BF16, INT8 变种)
```

---

## 问题 2: 生成的 Kernels 包含不同的量化模式

### 发现

我们提取的 12 个 kernel 文件包含 **两种不同的量化模式**：

#### 2.1 PER_COLUMN_SCALE_ONLY (4 个文件)
```bash
$ grep "PER_COLUMN_SCALE_ONLY" generated_kernels_w4a16_only/*.cu
cutlass_kernel_file_gemm_sm90_M128_group11.generated.cu
cutlass_kernel_file_gemm_sm90_M128_group14.generated.cu
cutlass_kernel_file_gemm_sm90_M128_group21.generated.cu
cutlass_kernel_file_gemm_sm90_M128_group25.generated.cu
```

特点:
- 每个输出通道 (N) 一个 scale
- Scales shape: `[N]`
- 适用于 per-channel 量化

#### 2.2 FINEGRAINED_SCALE_ONLY (8 个文件)
```bash
$ grep "FINEGRAINED_SCALE_ONLY" generated_kernels_w4a16_only/*.cu
cutlass_kernel_file_gemm_sm90_M128_group12.generated.cu
cutlass_kernel_file_gemm_sm90_M128_group13.generated.cu
cutlass_kernel_file_gemm_sm90_M128_group22.generated.cu
cutlass_kernel_file_gemm_sm90_M128_group23.generated.cu
cutlass_kernel_file_gemm_sm90_M128_group24.generated.cu
cutlass_kernel_file_gemm_sm90_M64_group11.generated.cu
cutlass_kernel_file_gemm_sm90_M64_group12.generated.cu
cutlass_kernel_file_gemm_sm90_M64_group6.generated.cu
```

特点:
- 每 `group_size` 个 K 一个 scale
- Scales shape: `[N, K/group_size]`
- 适用于 group-wise 量化 (更精确)

### 问题

我们的测试程序使用 `FINEGRAINED_SCALE_ONLY`，但：
1. 只有 **8 个文件** 包含这种模式
2. 这 8 个文件的配置可能不是最优的
3. 我们没有启用 per-column 模式的 kernels

---

## 问题 3: Kernel 配置不完整

### TensorRT-LLM 的完整 Kernel 生成流程

#### 3.1 生成阶段

```bash
# TensorRT-LLM CMake 配置时执行
python3 generate_kernels.py -a "90" -o ${INSTANTIATION_GENERATION_DIR}/gemm
```

这会生成 **所有架构 (SM80, SM90)** 的 kernels:
- 48 个 SM90 文件 (我们过滤后得到 12 个)
- N 个 SM80 文件
- 包含多种量化模式
- 包含多种 tile 配置

#### 3.2 链接阶段

```cmake
# cpp/tensorrt_llm/kernels/cutlass_kernels/CMakeLists.txt:200
add_instantiations(fpA_intB_gemm_src ${INSTANTIATION_GENERATION_DIR}/gemm)
```

`add_instantiations` 函数会：
```cmake
function(add_instantiations library base_dir)
  macro(glob_src_create_target ARCH BUILD_ARCHS)
    file(GLOB_RECURSE INSTANTIATIONS_GENERATED_${ARCH} ${base_dir}/${ARCH}/*.cu)
    # ...
    add_library(${TARGET_NAME} OBJECT ${INSTANTIATIONS_GENERATED_${ARCH}})
    target_link_libraries(${library} PRIVATE ${TARGET_NAME})
  endmacro()

  glob_src_create_target(80 "80;86;90;100f;120f")  # SM80 kernels
  glob_src_create_target(90 90)                     # SM90 kernels
  # ...
endfunction()
```

这说明：
1. TensorRT-LLM 会链接 **所有生成的 SM90 kernels** (不只是 W4A16)
2. 还会链接 SM80 kernels 作为 fallback
3. 我们只提取了 12 个 W4A16 文件，**缺少其他配置**

#### 3.3 编译选项

```cmake
# SM90 编译时
if(${ARCH} EQUAL 90)
    process_target(${TARGET_NAME} true false)
endif()

# process_target 会设置:
target_compile_definitions(${target_name} PUBLIC COMPILE_HOPPER_TMA_GEMMS)
target_compile_definitions(${target_name} PUBLIC COMPILE_HOPPER_TMA_GROUPED_GEMMS)
```

我们的构建脚本 **已经包含** 这些定义，这部分是正确的。

---

## 问题 4: 缺少 Tile 配置

### 查看生成的 kernel 配置

以 `cutlass_kernel_file_gemm_sm90_M128_group12.generated.cu` 为例:

```cpp
// 包含的 tile 配置:
cute::Shape<cute::Int<128>, cute::Int<64>, cute::Int<64>>   // M=128, N=64, K=64
cute::Shape<cute::Int<128>, cute::Int<128>, cute::Int<64>>  // M=128, N=128, K=64
cute::Shape<cute::Int<128>, cute::Int<256>, cute::Int<64>>  // M=128, N=256, K=64

// Cluster shapes:
ClusterShape::ClusterShape_1x1x1
ClusterShape::ClusterShape_2x1x1
ClusterShape::ClusterShape_1x2x1
ClusterShape::ClusterShape_2x2x1

// Mainloop schedules:
KernelTmaWarpSpecializedCooperative
```

### 问题

1. **M 维度有限**: 只有 M=64 和 M=128 的 tile
   - 对于 M < 64，性能会很差 (小 batch 场景)
   - 对于 M > 128，可能也不是最优

2. **缺少其他 tile 组合**:
   - 没有 M=256 的配置
   - 没有 K=128 的配置
   - N 维度选择有限

3. **缺少 Pingpong schedule**:
   - 只有 Cooperative schedule
   - Pingpong 在某些情况下性能更好

---

## 性能差的根本原因

### 总结

我们提取的 kernels 性能差的**根本原因**是:

1. ❌ **缺少 CUDA Core GEMV kernels**
   - 对于小 batch (M < 64)，CUDA Core kernels 性能远优于 CUTLASS
   - TensorRT-LLM 会自动选择，但我们没有包含

2. ❌ **Kernel 配置不完整**
   - 只有 12 个文件，缺少很多 tile 配置
   - TensorRT-LLM 有更多候选配置可供选择

3. ❌ **缺少动态选择机制**
   - TensorRT-LLM 使用 profiling 在运行时选择最优 kernel
   - 我们只是使用 `getConfigs()[0]`，可能不是最优

4. ❌ **可能缺少 SM80 fallback**
   - 某些操作可能需要 SM80 kernels
   - 我们完全移除了 SM80 支持

---

## 验证假设

### 实验 1: 检查 TensorRT-LLM 在运行时使用哪个 kernel

在原始 TensorRT-LLM 中添加日志:

```cpp
// fpA_intB_gemm_template.h 的 gemm() 函数中
printf("Selected config: tile=%s, mainloop=%d, cluster=%s, enableCudaKernel=%d\n",
       gemm_config.tile_config_str.c_str(),
       static_cast<int>(gemm_config.mainloop_schedule),
       gemm_config.cluster_shape_str.c_str(),
       gemm_config.enableCudaKernel);
```

预期发现:
- 对于小 batch (M=1-16): `enableCudaKernel=true` (使用 CUDA Core)
- 对于大 batch (M>64): `enableCudaKernel=false` (使用 CUTLASS)

### 实验 2: 对比性能

| Kernel Type | M=1 | M=16 | M=64 | M=256 | M=1024 |
|-------------|-----|------|------|-------|--------|
| CUDA Core GEMV | ✅ 快 | ✅ 快 | ? | ❌ 慢 | ❌ 慢 |
| CUTLASS TMA | ❌ 慢 | ❌ 慢 | ✅ 快 | ✅ 快 | ✅ 快 |

---

## 解决方案

### 方案 1: 添加 CUDA Core GEMV Kernels (推荐)

**步骤**:

1. 包含 `weightOnlyBatchedGemv` 目录:
   ```cmake
   # 在 build_w4a16_only.sh 中添加
   file(GLOB GEMV_SOURCES
       "${TRTLLM}/cpp/tensorrt_llm/kernels/weightOnlyBatchedGemv/*.cu"
   )

   add_executable(w4a16_only_test
       w4a16_minimal_test.cu
       ${W4A16_KERNELS}
       ${GEMV_SOURCES}  # ← 添加 GEMV kernels
       ${COMMON_SOURCES}
   )
   ```

2. 在测试程序中启用 CUDA kernel:
   ```cpp
   // 获取所有配置
   auto configs = runner.getConfigs();

   // 根据 M 大小选择
   CutlassGemmConfig gemm_config;
   if (M < 64) {
       // 选择 enableCudaKernel=true 的配置
       for (auto& cfg : configs) {
           if (cfg.enableCudaKernel) {
               gemm_config = cfg;
               break;
           }
       }
   } else {
       // 使用默认 CUTLASS 配置
       gemm_config = configs[0];
   }
   ```

**优点**:
- ✅ 小 batch 性能大幅提升
- ✅ 覆盖完整的 batch 范围
- ✅ 与 TensorRT-LLM 行为一致

**缺点**:
- 需要编译更多源文件
- 增加可执行文件大小

### 方案 2: 生成并包含所有 Kernel 配置

**步骤**:

1. 重新生成 kernels，包含所有配置:
   ```bash
   # 不过滤，保留所有 48 个 SM90 kernels
   python3 generate_kernels.py -a "90" -o /tmp/w4a16_all
   cp /tmp/w4a16_all/gemm/90/*.cu generated_kernels_all_sm90/
   ```

2. 修改过滤逻辑，只移除 FP8 (保留 BF16):
   ```bash
   # 只移除 FP8，保留 BF16
   for f in *.cu; do
       if ! grep -q "__nv_fp8" "$f"; then
           cp "$f" generated_kernels_no_fp8/
       fi
   done
   ```

**优点**:
- ✅ 更多 tile 配置可供选择
- ✅ 支持 BF16 (如果需要)

**缺点**:
- 编译时间更长
- 可执行文件更大
- 仍然缺少 CUDA Core kernels

### 方案 3: 实现 Kernel Profiling (最完整)

模仿 TensorRT-LLM 的做法:

```cpp
// 首次运行时 profile 所有配置
auto configs = runner.getConfigs();
float best_time = FLT_MAX;
CutlassGemmConfig best_config;

for (auto& cfg : configs) {
    // 预热
    runner.gemm(..., cfg, ...);
    cudaDeviceSynchronize();

    // 测速
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        runner.gemm(..., cfg, ...);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);

    if (time_ms < best_time) {
        best_time = time_ms;
        best_config = cfg;
    }
}

// 使用最优配置
runner.gemm(..., best_config, ...);
```

**优点**:
- ✅ 找到当前硬件的最优配置
- ✅ 适应不同的矩阵尺寸

**缺点**:
- 首次运行慢 (需要 profiling)
- 需要实现配置缓存

---

## 推荐方案

### 综合推荐: 方案 1 + 部分方案 3

1. **添加 CUDA Core GEMV kernels** (方案 1)
   - 解决小 batch 性能问题
   - 这是最关键的缺失部分

2. **实现简单的配置选择** (方案 3 的简化版)
   ```cpp
   auto configs = runner.getConfigs();
   CutlassGemmConfig gemm_config;

   // 根据 M 大小简单选择
   if (M < 64) {
       // 寻找 CUDA kernel
       for (auto& cfg : configs) {
           if (cfg.enableCudaKernel) {
               gemm_config = cfg;
               break;
           }
       }
   } else {
       // 对于大 M，遍历找到最合适的 tile
       for (auto& cfg : configs) {
           if (!cfg.enableCudaKernel &&
               cfg.tile_config.m >= M/4 &&  // 启发式选择
               cfg.tile_config.m <= M) {
               gemm_config = cfg;
               break;
           }
       }
   }

   if (gemm_config.tile_config.m == 0) {
       gemm_config = configs[0];  // Fallback
   }
   ```

3. **可选: 增加更多 kernel 文件**
   - 如果编译时间可接受，包含更多 tile 配置

---

## 技术细节

### CUDA Core GEMV vs CUTLASS TMA

| 特性 | CUDA Core GEMV | CUTLASS TMA |
|------|----------------|-------------|
| **指令类型** | FMA (标量) | WGMMA (Tensor Core) |
| **内存访问** | 普通 load/store | TMA (异步, 批量) |
| **最优场景** | M < 64 (小 batch) | M > 64 (大 batch) |
| **延迟** | 低 | 中等 |
| **吞吐** | 低-中 | 高 |
| **寄存器压力** | 低 | 高 |
| **Occupancy** | 高 | 中等 |

### 为什么小 batch 用 CUDA Core 更快？

1. **Tensor Core 启动开销**:
   - WGMMA 需要足够的数据才能充分利用
   - 小 M 时，很多 Tensor Core 空闲

2. **TMA 开销**:
   - TMA 有固定的启动延迟
   - 小矩阵时，延迟占比太大

3. **GEMV 特化**:
   - M=1 时是向量-矩阵乘法
   - CUDA Core 可以更灵活地处理

### 为什么大 batch 用 CUTLASS 更快？

1. **Tensor Core 吞吐**:
   - WGMMA 吞吐是 FMA 的 8-16 倍
   - 大 M 时可以充分利用

2. **TMA 批量传输**:
   - 可以一次传输大块数据
   - Amortize 启动开销

3. **更好的 cache 利用**:
   - Cluster 可以共享 L2 cache
   - 减少 DRAM 访问

---

## 文件清单

### 当前包含 (性能差)
```
├── generated_kernels_w4a16_only/     # 12 个 CUTLASS kernels
│   ├── M128_group11-14.cu (4 files) - PER_COLUMN
│   ├── M128_group21-25.cu (5 files) - FINEGRAINED (部分)
│   └── M64_group6,11,12.cu (3 files) - FINEGRAINED (部分)
├── build_w4a16_only.sh
└── w4a16_minimal_test.cu
```

### 需要添加 (提升性能)
```
├── cpp/tensorrt_llm/kernels/weightOnlyBatchedGemv/   # ← 关键!
│   ├── cudaCoreGemm.cu                               # 核心 GEMV
│   ├── kernelDispatcherFp16Int4*.cu                  # FP16+INT4 dispatchers
│   └── ...
└── [可选] 更多 generated kernels                      # 更多 tile 配置
```

---

## 总结

### 问题根源

1. **缺少 CUDA Core GEMV kernels** - 这是性能差的主要原因
2. Kernel 配置不完整 - 只有 12 个文件
3. 缺少动态选择机制 - 总是用第一个配置

### 解决方案优先级

1. **P0 (必须)**: 添加 `weightOnlyBatchedGemv` kernels
2. **P1 (推荐)**: 实现简单的配置选择逻辑
3. **P2 (可选)**: 包含更多 CUTLASS kernel 配置
4. **P3 (可选)**: 实现完整的 profiling 机制

### 预期提升

实现方案 1 后:
- 小 batch (M < 64): **10-50x** 性能提升
- 中 batch (M = 64-256): **2-5x** 性能提升
- 大 batch (M > 256): **1.5-2x** 性能提升

---

**Date**: 2025-11-15
**Branch**: w4a16_integration
**Status**: 分析完成，等待实施
