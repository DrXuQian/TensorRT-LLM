# W4A16 SM90 Kernels Extraction

从 TensorRT-LLM 中提取的纯 W4A16 (FP16+INT4) Hopper kernels，无 FP8/BF16 依赖。

## 快速开始 (H800)

```bash
# 克隆并构建
git clone https://github.com/DrXuQian/TensorRT-LLM.git
cd TensorRT-LLM
git checkout w4a16_integration
./build_w4a16_only.sh

# 运行测试
./build_w4a16_only/w4a16_only_test 512 1024 1024
```

## 内容

- **12 个 W4A16 kernel 文件** - 纯 FP16+INT4，无 FP8/BF16
- **构建脚本** - `build_w4a16_only.sh`
- **测试程序** - `w4a16_minimal_test.cu`

## 文档

- [README_W4A16_EXTRACTION.md](README_W4A16_EXTRACTION.md) - 提取过程详解
- [H800_QUICKSTART.md](H800_QUICKSTART.md) - H800 快速开始
- [W4A16_SM90_SUCCESS.md](W4A16_SM90_SUCCESS.md) - 技术文档

## 提取方法

1. 生成 48 个 SM90 kernel 文件 (Python 脚本)
2. 过滤出 12 个纯 FP16+INT4 文件
3. 使用正确的编译选项构建

详见 [README_W4A16_EXTRACTION.md](README_W4A16_EXTRACTION.md)

## GPU 要求

**仅支持 H100/H800** (Compute Capability 9.0)
- 使用 TMA 和 WGMMA 指令
- 其他 GPU 需要不同架构的 kernels

## 特性

✅ 无 FP8 - 移除所有 `__nv_fp8` 类型
✅ 无 BF16 - 移除所有 `__nv_bfloat16` 类型
✅ 纯 W4A16 - FP16 激活 + INT4 权重
✅ SM90 优化 - TMA + WGMMA

---

**Branch**: `w4a16_integration`
**Author**: Claude Code
**Date**: 2025-11-15
