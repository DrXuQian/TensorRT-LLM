# W4A16 Hopper Kernel 提取总结

**日期**: 2025-11-14
**最终结论**: 独立提取需要生成 kernel 实例化

---

## 核心问题

W4A16 SM90 kernels 的实例化**不在源代码中**，而是通过 Python 脚本在构建时生成。

### 链接错误

```
undefined reference to `void tensorrt_llm::kernels::cutlass_kernels_oss::
sm90_generic_mixed_gemm_kernelLauncher<...>(...)'
```

### 根本原因

1. **模板实现**: `fpA_intB_gemm_template_sm90.h` 包含所有模板代码（这是头文件）
2. **显式实例化**: `fp16_int4_gemm_fg_scaleonly.cu` 只实例化 `CutlassFpAIntBGemmRunner`
3. **Kernel launcher 实例化**: 在 **构建时** 由 `generate_kernels.py` 生成到 `build/cutlass_instantiations/`

---

## 测试结果

### ✅ 编译成功
- 模板头文件可以编译
- `CutlassFpAIntBGemmRunner` 可以实例化
- CUTLASS 确实是纯头文件库

### ❌ 链接失败
- 缺少大量 `sm90_generic_mixed_gemm_kernelLauncher` 实例化
- 这些实例化对应不同的 CTA shapes (64x16x64, 64x32x64, 128x16x64, 等等)
- 每个配置都需要单独的实例化

---

## 为什么这样设计？

### 编译时间优化

如果把所有可能的 kernel 配置都写在源码中实例化：
- **数十个** CTA shape 组合
- **多种** activation types (FP16, BF16)
- **多种** weight types (INT4, INT8)
- **多种** epilogues (bias, no bias, etc.)

总计 **数百个** kernel 变体！全部实例化会导致：
- 极长的编译时间（几小时）
- 巨大的二进制文件（几 GB）

### 动态生成方案

TensorRT-LLM 的做法：
1. CMake 检测 CUDA architecture (SM90)
2. Python 脚本根据架构生成**需要的**实例化
3. 只编译当前硬件需要的 kernel

---

## 三种解决方案

### 方案 1: 使用 TensorRT-LLM Python API ⭐⭐⭐⭐⭐

**最简单、最推荐**

```python
import tensorrt_llm
# TensorRT-LLM 自动处理一切
```

**优点**:
- 零配置
- 自动选择最优 kernel
- 官方支持
- 完整功能

**缺点**:
- 需要安装完整 TensorRT-LLM

---

### 方案 2: 在 TensorRT-LLM 构建系统内工作 ⭐⭐⭐⭐

**适合需要修改或研究 kernel 的情况**

```bash
# 在 TensorRT-LLM 目录中
mkdir build && cd build
cmake -DBUILD_PYT=ON -DCMAKE_CUDA_ARCHITECTURES=90 ..
make -j
```

这会：
1. 运行 `generate_kernels.py` 生成实例化
2. 编译所有需要的 kernels
3. 生成 Python bindings

然后可以：
- 修改 kernel 实现
- 添加新的 epilogues
- 测试性能

**优点**:
- 可以修改源码
- 完整的开发环境
- 正确的构建流程

**缺点**:
- 需要完整依赖
- 构建时间较长

---

### 方案 3: 手动创建实例化文件 ⭐⭐

**最复杂，但可以独立提取**

需要创建类似这样的文件：

```cpp
// w4a16_sm90_instances.cu
#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm_template_sm90.h"

namespace tensorrt_llm {
namespace kernels {
namespace cutlass_kernels_oss {

// 为每个需要的配置创建实例化
template void sm90_generic_mixed_gemm_kernelLauncher<
    half, cutlass::uint4b_t, half, half, half,
    cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY,
    tensorrt_llm::cutlass_extensions::EpilogueOpBias,
    cute::tuple<cute::C<64>, cute::C<16>, cute::C<64>>,
    cute::tuple<cute::C<1>, cute::C<1>, cute::C<1>>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized
>(...);

// 重复几十次，每个 CTA shape 一个...

}}}
```

**需要实例化的配置** (从链接错误中提取):
- 64x16x64, 64x32x64, 64x64x64, 64x128x64, 64x256x64 (Pingpong)
- 128x16x64, 128x32x64, 128x64x64, 128x128x64, 128x256x64 (Cooperative)
- 每个还有多种 cluster shapes...

**优点**:
- 可以独立于 TensorRT-LLM 构建
- 完全控制包含哪些 kernels

**缺点**:
- 需要手动维护几十个实例化
- 容易出错
- 需要深入理解 CUTLASS 模板参数

---

## 推荐方案对比

| 需求 | 推荐方案 | 理由 |
|------|----------|------|
| 使用 W4A16 inference | 方案 1 (Python API) | 最简单 |
| 研究 kernel 实现 | 阅读源码 + 方案 2 | 完整环境 |
| 修改/优化 kernels | 方案 2 (TensorRT-LLM build) | 正确的开发流程 |
| 性能测试 | 方案 2 | 可以测试所有配置 |
| 独立库提取 | 方案 3 (手动实例化) | 可行但复杂 |

---

## 如果必须独立提取

### 步骤

1. **运行 generate_kernels.py**:
```bash
cd /home/qianxu/TensorRT-LLM/cpp/tensorrt_llm/kernels/cutlass_kernels/python
python3 generate_kernels.py -a "90" -o /tmp/instantiations
```

2. **复制生成的文件**:
```bash
cp /tmp/instantiations/gemm/*.cu /home/qianxu/my_project/
```

3. **添加到编译**:
```cmake
add_executable(my_test
    my_test.cu
    ${GENERATED_INSTANTIATIONS}/*.cu
    ...
)
```

### 问题

- Python 脚本可能依赖 CUTLASS Python library
- 生成的代码可能有额外依赖
- 需要手动处理 API 变化

---

## 文件位置总结

| 类型 | 位置 | 说明 |
|------|------|------|
| 模板实现 | `fpA_intB_gemm_template_sm90.h` | 纯头文件，可直接使用 |
| Launcher实现 | `launchers/fpA_intB_launcher_sm90.inl` | 头文件，包含实际 kernel 调用 |
| Runner 实例化 | `fp16_int4_gemm_fg_scaleonly.cu` | 显式实例化 Runner class |
| Kernel 实例化 | `build/cutlass_instantiations/gemm/*.cu` | **构建时生成** |
| 生成脚本 | `python/generate_kernels.py` | Python 脚本 |

---

## 最终建议

根据你的实际需求：

1. **如果只是想用 W4A16**: 用 TensorRT-LLM Python API
2. **如果想学习实现**: 阅读源码就够了
3. **如果想修改优化**: Fork TensorRT-LLM，在其构建系统内工作
4. **如果必须独立提取**: 考虑只提取 Ampere/Ada 版本（不需要 TMA，更简单）

---

## Ampere/Ada 版本的优势

`fpA_intB_gemm_template.h` (非 SM90 版本):
- ✅ 不使用 TMA
- ✅ 实例化更少更简单
- ✅ 可在 RTX 3090/4090 等 GPU 上运行
- ✅ 仍然是 W4A16 量化
- ✅ 性能仍然很好（只是比 Hopper 慢一些）

---

**结论**: W4A16 Hopper kernel 确实可以提取，但需要理解并处理构建时的 kernel 生成机制。对于大多数用例，直接使用 TensorRT-LLM 的 API 是最佳选择。

---

**作者**: Claude Code
**日期**: 2025-11-14
