# W4A16 Hopper Kernel 提取状态报告

**日期**: 2025-11-14
**位置**: `/home/qianxu/trt_llm_w4a16_hopper/`
**GitHub**: https://github.com/DrXuQian/TensorRT-LLM/tree/w4a16_hopper_extraction

---

## ✅ 已完成的工作

### 1. 成功提取 Hopper Kernel
- ✅ 从 TensorRT-LLM 完整提取 W4A16 Hopper (SM90) kernel
- ✅ 包含 72 个 CUTLASS extension 文件
- ✅ 包含完整的 launcher 和 template 实现
- ✅ 包含所有必要的头文件和依赖

### 2. 成功编译
- ✅ 编译生成 `libw4a16_sm90_kernel.so` (2.7 MB)
- ✅ 包含两个 kernel 变体:
  - `w4a16_sm90_gemm_128`: 128×128×128 CTA, TMA Cooperative
  - `w4a16_sm90_gemm_64`: 64×128×128 CTA, TMA Pingpong
- ✅ 编译了 6 个测试/benchmark 程序

### 3. 验证 Kernel 编译
```bash
$ cuobjdump -symbols lib/libw4a16_sm90_kernel.so | grep kernel
# 确认包含:
- MainloopSm90TmaGmmaRmemAWarpSpecializedMixedInput (Pingpong)
- MainloopSm90TmaGmmaRmemAWarpSpecializedMixedInput (Cooperative)
```

### 4. Git 历史记录
完整的提取过程已记录在 10 个 git commits 中，便于追溯。

---

## ❌ 遇到的问题

### 运行时问题 (H800 测试)

**症状**:
1. 所有测试程序都在 `cudaStreamSynchronize()` 时 segfault
2. `ncu` 看不到完整的 kernel 执行
3. 输出全为 0（因为 kernel 没完成）

**诊断结果**:
```
✅ Kernel call returned successfully!
✅ No kernel launch errors
  Synchronizing stream...
Segmentation fault (core dumped)
```

**结论**:
- Kernel **成功启动**了
- 但在 **GPU 执行过程中崩溃**
- 问题在 kernel 内部，不是启动问题

---

## 🔍 问题根本原因分析

### 可能原因 1: INT4 数据格式问题

当前代码使用的 INT4 打包格式可能与 kernel 预期不符。

**当前实现**:
```cpp
// 每个字节存储 2 个 INT4 值
uint8_t val1 = dis(gen) & 0xF;  // 低 4 位
uint8_t val2 = dis(gen) & 0xF;  // 高 4 位
data[i] = (val2 << 4) | val1;
```

**问题**: CUTLASS 可能期望不同的交错（interleaved）格式。

### 可能原因 2: Bias 处理问题

Launcher 代码 (fpA_intB_launcher_sm90.inl:234-246):
```cpp
// Line 235: 用 output C 作为 bias 的占位符
auto output_as_bias_type = reinterpret_cast<CutlassBiasType const*>(C);

// Line 241: 在构造 Gemm::Arguments 时使用
{{}, output_as_bias_type, stride_D, ...}

// Line 246: 在 epilogue.thread 中使用真实 bias
{reinterpret_cast<CutlassBiasType const*>(biases), CutlassBiasType(0.f)}
```

**问题**: 这个双重设置可能导致 TMA descriptor 混淆。

### 可能原因 3: Group Size 与 CTA K 维度不匹配

```cpp
// Line 193-198: group_size 必须是 cta_shape_k 的倍数
if (group_size % cta_shape_k != 0) {
    throw std::runtime_error("The group size must a multiple of 128");
}
```

**当前测试**: group_size=128, CTA K=128 ✓ (应该没问题)

### 可能原因 4: TMA 需要特定的内存对齐

TMA 对内存对齐有严格要求，我们使用 `cudaMemset` 初始化可能不满足。

---

## 📁 项目结构

```
/home/qianxu/trt_llm_w4a16_hopper/
├── build/
│   ├── lib/libw4a16_sm90_kernel.so     # 2.7 MB
│   └── bin/
│       ├── test_w4a16_sm90             # 简单测试
│       ├── benchmark_w4a16_sm90        # 完整 benchmark
│       ├── debug_w4a16_sm90            # 调试版本
│       ├── simple_test_sm90            # 简化测试
│       ├── test_with_bias              # 带 bias 测试
│       └── safe_test                   # 异常捕获测试
├── src/
│   ├── w4a16_sm90_kernel.cu            # Kernel wrappers
│   ├── benchmark_w4a16.cu              # 性能测试
│   ├── debug_kernel.cu                 # 调试程序
│   ├── simple_test.cu                  # 简单测试
│   ├── test_with_bias.cu               # Bias 测试
│   ├── safe_test.cu                    # 安全测试
│   ├── logger.cpp                      # TensorRT-LLM logger
│   ├── stringUtils.cpp                 # 字符串工具
│   ├── assert.cpp                      # 断言实现
│   └── tllmException.cpp               # 异常处理
├── include/
│   └── tensorrt_llm/
│       ├── common/                     # 通用头文件
│       ├── cutlass_extensions/         # 72 个 CUTLASS 扩展
│       └── kernels/                    # Kernel 头文件
├── CMakeLists.txt                      # 构建配置
├── README.md                           # 项目说明
├── BUILD_SUCCESS.md                    # 构建文档
├── QUICKSTART.md                       # 快速开始
├── IMPORTANT_NOTICE.md                 # 重要说明
└── STATUS_REPORT.md                    # 本文件
```

---

## 🚀 下一步建议

### 选项 A: 修复当前 Hopper Kernel (困难)

需要深入调试，可能需要:
1. 使用 `cuda-gdb` 在 GPU 上调试
2. 修复 INT4 数据格式
3. 修复 bias/TMA descriptor 处理
4. 可能需要修改 CUTLASS 模板参数

**难度**: ⭐⭐⭐⭐⭐
**时间**: 数天到数周
**成功率**: 中等

### 选项 B: 提取 Ampere/Ada 版本 (推荐) ⭐

提取 TensorRT-LLM 中的 Ampere (SM80) / Ada (SM89) 版本的 W4A16 kernel。

**优点**:
- 不使用 TMA，更容易调试
- 可以在 RTX 3090/4090/5070 等 GPU 上运行
- 仍然是 W4A16 量化
- 代码更成熟，已在生产环境验证

**位置**: `cpp/tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm_template.h`

**难度**: ⭐⭐
**时间**: 1-2 天
**成功率**: 高

### 选项 C: 使用 TensorRT-LLM Python API

直接使用 TensorRT-LLM 的 Python API 调用 W4A16 kernel，无需提取。

**优点**:
- 开箱即用
- 完整的功能支持
- 官方维护

**缺点**:
- 需要整个 TensorRT-LLM 环境
- 不是独立的 kernel

---

## 📊 性能对比 (预期)

| 版本 | GPU | TMA | 性能 | 稳定性 |
|------|-----|-----|------|--------|
| Hopper (当前) | H100/H200 | ✅ | 最高 | ❌ 崩溃 |
| Ampere/Ada | RTX 3090/4090 | ❌ | 高 | ✅ 稳定 |
| Blackwell | RTX 5070 | ? | 未知 | ❓ 未测试 |

---

## 📝 技术细节

### Kernel 配置

**W4A16 量化**:
- Weights: 4-bit integer (INT4)
- Activations: 16-bit floating point (FP16/BF16)
- Quantization: Fine-grained, group-wise
- Group size: 128 (可配置)

**CTA 配置**:
- 大矩阵: 128×128×128 with TMA Cooperative
- 小矩阵: 64×128×128 with TMA Pingpong

**编译选项**:
- Target: SM90 (Hopper)
- CUDA: 12.8
- Flag: `-DCOMPILE_HOPPER_TMA_GEMMS`

### 依赖

- CUTLASS (从 TensorRT-LLM 3rdparty)
- CUDA Toolkit 12.x
- CMake 3.18+
- C++17

---

## 🎯 总结

✅ **成功**: 完整提取并编译了 W4A16 Hopper kernel
❌ **失败**: Kernel 在 H800 运行时崩溃
🔧 **原因**: Kernel 内部访问无效内存（可能是数据格式或 TMA descriptor 问题）
💡 **建议**: 提取 Ampere/Ada 版本，更容易使用且稳定

---

**状态**: 编译成功，运行失败
**优先级**: 建议转向 Ampere/Ada 版本
**文档完整性**: ✅ 完整
**代码可用性**: ⚠️ 需要在真实 Hopper GPU (H100) 上验证
