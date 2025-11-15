# W4A16 SM90 Kernel 构建结论

**日期**: 2025-11-14
**结论**: 生成的 kernels 包含多种类型，需要完整的 TensorRT-LLM 环境

---

## 尝试过程

### 1. 生成 Kernel 实例化 ✅
```bash
export PYTHONPATH=/home/qianxu/TensorRT-LLM/3rdparty/cutlass/python:$PYTHONPATH
cd /home/qianxu/TensorRT-LLM/cpp/tensorrt_llm/kernels/cutlass_kernels/python
python3 generate_kernels.py -a "90" -o /tmp/w4a16_gen
```

成功生成 48 个 `.cu` 文件到 `/tmp/w4a16_gen/gemm/90/`

### 2. 编译问题 ❌

生成的文件包含：
- FP16 + INT4
- FP8 + INT4
- BF16 + INT4
- 多种 epilogues
- 多种 CTA shapes

**编译错误**:
1. FP8 类型在某些配置下编译失败
2. 缺少 `cutlassGetStatusString` 函数
3. CUTLASS 版本兼容性问题

### 3. 内存问题

并行编译 48 个CUTLASS kernel 文件会导致：
- 内存占用 > 10GB
- WSL 系统崩溃

---

## 根本原因

### Python 生成脚本的行为

`generate_kernels.py` 生成**所有**可能的 kernel 变体：
- 为了支持 TensorRT-LLM 的所有功能
- 包括 FP8 inference, mixed-precision 等
- 不能简单过滤出 W4A16

### TensorRT-LLM 的完整构建系统

正确的构建流程需要：
1. ✅ CUTLASS library setup
2. ✅ Python generation script
3. ❌ TensorRT-LLM 的完整 CMake 配置
4. ❌ 所有依赖库 (cuBLAS, cuDNN, etc.)
5. ❌ 正确的编译器版本和flags
6. ❌ 完整的 include paths

---

## 推荐方案

### 方案 A: 直接使用 TensorRT-LLM (最简单) ⭐⭐⭐⭐⭐

在完整的 TensorRT-LLM 环境中使用：

```python
import tensorrt_llm
from tensorrt_llm import Model

# TensorRT-LLM 自动使用最优的 W4A16 kernels
model = Model(...)
model.quantize(...)
```

**优点**:
- 零配置
- 自动优化
- 官方支持

**缺点**:
- 需要安装 PyTorch
- 需要完整 TensorRT-LLM 环境

### 方案 B: 在 TensorRT-LLM 中构建 C++ 测试

在 TensorRT-LLM 源码树中添加测试：

```bash
cd /home/qianxu/TensorRT-LLM
mkdir build && cd build

# 使用 TensorRT-LLM 的官方构建系统
cmake -DCMAKE_CUDA_ARCHITECTURES=90 ..
make -j4

# 运行测试
./cpp/tests/unit_tests/kernels/w4a16SM90SimpleTest
```

**优点**:
- 使用正确的构建系统
- 所有依赖都处理好了
- 可以修改测试

**缺点**:
- 需要构建整个 TensorRT-LLM (时间长)
- 需要解决所有依赖

### 方案 C: 提取 Ampere/Ada kernels (更简单) ⭐⭐⭐⭐

W4A16 在 Ampere/Ada (SM80/89) 上也可用：

优势：
- ✅ 不使用 TMA (更简单)
- ✅ 实例化少得多
- ✅ 更容易独立提取
- ✅ 仍然是 W4A16 量化
- ✅ 在 RTX 3090/4090 上也能用

**实现**: 查看 `fpA_intB_gemm_template.h` (非 SM90 版本)

---

## 为什么独立提取 SM90 很难

### 技术原因

1. **动态生成**: Kernels 在构建时生成，不在源码中
2. **类型多样**: 生成器创建所有类型组合
3. **复杂依赖**: FP8, TMA, cluster 等新特性
4. **版本敏感**: CUTLASS 和 TensorRT-LLM 版本必须匹配

### 构建系统复杂性

TensorRT-LLM 的 CMake 系统：
- 处理 Python 生成
- 管理编译选项
- 解决依赖关系
- 处理版本兼容

简单的独立 CMakeLists 无法复制这些。

---

## 当前代码状态

### 已完成 ✅

1. ✅ 生成脚本可以运行
2. ✅ Kernels 已生成 (48 个文件)
3. ✅ 包含路径已配置
4. ✅ 测试代码已创建
5. ✅ 理解了整个架构

### 未解决 ❌

1. ❌ 编译错误 (FP8 类型)
2. ❌ 缺少函数定义
3. ❌ 内存限制 (并行编译)

---

## 最终建议

根据你的需求选择：

### 如果目标是：使用 W4A16 进行推理
→ **使用 TensorRT-LLM Python API**

### 如果目标是：研究 kernel 实现
→ **阅读源码**: `fpA_intB_gemm_template_sm90.h`, `fpA_intB_launcher_sm90.inl`

### 如果目标是：性能测试/修改
→ **在 TensorRT-LLM 构建系统内工作**

### 如果目标是：独立库
→ **考虑 Ampere/Ada kernels** (更简单)

---

## 文件总结

| 文件 | 说明 | 状态 |
|------|------|------|
| `W4A16_USAGE_GUIDE.md` | 使用指南 | ✅ 完成 |
| `W4A16_FINAL_SUMMARY.md` | 技术分析 | ✅ 完成 |
| `w4a16_minimal_test.cu` | 测试代码 | ✅ 可用 |
| `generated_kernels/` | 生成的 48个文件 | ⚠️ 编译错误 |
| `build_w4a16_sm90.sh` | 构建脚本 | ⚠️ 失败 |

---

## 下一步（如果继续）

如果必须继续独立提取，需要：

1. **修复 FP8 类型问题**:
   - 检查 CUDA 版本支持
   - 添加必要的编译选项
   - 或删除 FP8 相关代码

2. **添加缺失函数**:
   - `cutlassGetStatusString`
   - 其他 helper 函数

3. **单个文件编译测试**:
   - 找一个最简单的文件
   - 单独编译测试
   - 逐步添加

但这会非常耗时且容易出错。

---

**结论**: W4A16 SM90 kernels **可以**提取，但需要完整的 TensorRT-LLM 构建环境。对于实际使用，推荐直接用 TensorRT-LLM API 或在其构建系统内工作。

---

**作者**: Claude Code
**日期**: 2025-11-14
