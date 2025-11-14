# W4A16 Hopper Kernel 使用指南

**日期**: 2025-11-14
**目的**: 说明如何正确使用 TensorRT-LLM 中的 W4A16 Hopper kernels

---

## ⚠️ 重要发现

之前在 `/home/qianxu/trt_llm_w4a16_hopper/` 中的独立提取方式是**错误的**：

### 错误的方式 ❌
```cpp
// 直接调用底层 launcher 函数
extern "C" void w4a16_sm90_gemm_128(...);
w4a16_sm90_gemm_128(d_A, d_B, ...);  // 导致 segfault!
```

**问题**:
- 缺少proper CUTLASS initialization
- TMA descriptors配置不正确
- 没有workspace management
- 结果: `cudaStreamSynchronize()` 时 segmentation fault

### 正确的方式 ✅
```cpp
// 使用 TensorRT-LLM 的 CutlassFpAIntBGemmRunner API
using namespace tensorrt_llm::kernels::cutlass_kernels;

CutlassFpAIntBGemmRunner<half, cutlass::uint4b_t,
    cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> runner;

// 获取workspace大小
size_t workspace_bytes = runner.getWorkspaceSize(M, N, K);
cudaMalloc(&workspace, workspace_bytes);

// 调用 GEMM
runner.gemm(
    d_A,                  // activations (FP16)
    d_B,                  // weights (INT4)
    d_scales,             // scales (FP16)
    nullptr,              // zero points (可选)
    nullptr,              // biases (可选)
    1.0f,                 // alpha
    d_C,                  // output (FP16)
    M, N, K,              // dimensions
    group_size,           // 量化group size
    gemm_config,          // kernel configuration
    workspace,            // workspace
    workspace_bytes,      // workspace size
    stream                // CUDA stream
);
```

---

## 如何在 TensorRT-LLM 中使用

### 方法 1: 使用 TensorRT-LLM Python API (推荐)

TensorRT-LLM 已经提供了完整的 Python 接口：

```python
import tensorrt_llm
from tensorrt_llm.models import ...

# TensorRT-LLM 会自动选择最优的 kernel
# 包括 W4A16 Hopper kernels (如果在 H100/H800 上)
```

**优点**:
- 开箱即用
- 自动选择最优kernel配置
- 完整的功能支持
- 官方维护和更新

### 方法 2: 在 TensorRT-LLM C++ 代码中使用

如果需要在 C++ 中使用，直接使用 `CutlassFpAIntBGemmRunner`:

**位置**: `cpp/tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h`

```cpp
#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"

// 在你的 kernel launcher 或 layer 实现中
auto runner = std::make_unique<CutlassFpAIntBGemmRunner<
    half, cutlass::uint4b_t,
    cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY>>();

// 使用 runner->gemm(...) 调用
```

---

## 为什么独立提取很困难

SM90 (Hopper) kernels 的实例化不在普通的 `.cu` 文件中，而是通过构建时的 Python 脚本生成：

1. **Kernel Generation**: `cpp/tensorrt_llm/kernels/cutlass_kernels/python/generate_kernels.py`
2. **Instantiation**: 生成到 `build/cutlass_instantiations/` 目录
3. **条件编译**: 只有在 CMake 检测到 SM90 架构时才生成

**需要的组件**:
- CUTLASS library setup (`setup_library.py`)
- Kernel generation script
- 正确的 CMake 配置
- 所有 TensorRT-LLM 依赖

这使得独立提取和编译非常复杂且容易出错。

---

## 建议的开发流程

### 场景 1: 研究 W4A16 kernel 实现

**目标**: 理解 kernel 如何工作

**方法**:
1. 阅读 `fpA_intB_gemm_template_sm90.h` - kernel 模板定义
2. 查看 `launchers/fpA_intB_launcher_sm90.inl` - launcher 实现
3. 研究 CUTLASS extensions in `cpp/tensorrt_llm/cutlass_extensions/`

### 场景 2: 性能测试 W4A16 kernels

**目标**: 测试不同配置的性能

**方法**:
1. 使用 TensorRT-LLM 提供的 benchmark 工具
2. 或者在 `cpp/tests/unit_tests/kernels/weightOnly/` 中添加测试
3. 使用 `CutlassFpAIntBGemmRunner` 的 `getConfigs()` 获取所有可用配置
4. 逐个测试不同配置的性能

### 场景 3: 修改或扩展 W4A16 kernels

**目标**: 添加新功能或优化

**方法**:
1. Fork TensorRT-LLM repository
2. 在 TensorRT-LLM 的构建系统内进行修改
3. 修改 `fpA_intB_gemm_template_sm90.h` 或 launcher
4. 使用 TensorRT-LLM 的构建系统编译和测试
5. 提交 Pull Request (如果有价值)

---

## 单元测试示例

我们已经在 TensorRT-LLM 中添加了一个单元测试:

**文件**: [cpp/tests/unit_tests/kernels/weightOnly/w4a16_sm90_simple_test.cu](cpp/tests/unit_tests/kernels/weightOnly/w4a16_sm90_simple_test.cu)

这展示了正确的API使用方式。

---

## 技术细节

### W4A16 量化

- **Weights**: 4-bit integer (INT4)
- **Activations**: 16-bit floating point (FP16/BF16)
- **Quantization**: Fine-grained, group-wise
- **Group size**: 可配置 (通常128)

### Kernel 变体

| CTA Shape | Mainloop | Epilogue | 用途 |
|-----------|----------|----------|------|
| 128×128×128 | TMA Warp Specialized Cooperative | TMA Warp Specialized Cooperative | 大矩阵 |
| 64×128×128 | TMA Warp Specialized Pingpong | TMA Warp Specialized | 小矩阵 |

### 内存布局

- **A (activations)**: Row-major FP16
- **B (weights)**: CUTLASS mixed GEMM layout (INT4, 需预处理)
- **Scales**: Per-group FP16
- **Output**: Row-major FP16

---

## 总结

❌ **不要**: 尝试独立提取和直接调用 launcher 函数
✅ **应该**: 使用 `CutlassFpAIntBGemmRunner` API

`CutlassFpAIntBGemmRunner` 封装了所有复杂性:
- ✅ Proper CUTLASS initialization
- ✅ TMA descriptor configuration
- ✅ Workspace management
- ✅ Kernel configuration selection
- ✅ 错误处理

这是 TensorRT-LLM 团队设计和测试的正确使用方式。

---

## 参考资料

- **TensorRT-LLM Repo**: https://github.com/NVIDIA/TensorRT-LLM
- **CUTLASS**: https://github.com/NVIDIA/cutlass
- **Our previous standalone attempt**: `/home/qianxu/trt_llm_w4a16_hopper/STATUS_REPORT.md`
- **GitHub branch**: `w4a16_integration`

---

**作者**: Claude Code
**最后更新**: 2025-11-14
