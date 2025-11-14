# W4A16 Hopper (SM90) Kernel - 编译成功 ✓

## 构建信息

- **构建日期**: 2025-11-14
- **目标架构**: NVIDIA Hopper (SM90)
- **兼容架构**: SM90+ (包括 RTX 5070 的 SM120)
- **CUDA 版本**: 12.8.93
- **CUTLASS 版本**: TensorRT-LLM 3rdparty bundled

## 生成的文件

### 1. 共享库
```
lib/libw4a16_sm90_kernel.so (2.7 MB)
```
这是包含 W4A16 Hopper kernel 的主要库文件。

### 2. 测试可执行文件
```
bin/test_w4a16_sm90 (19 KB)
```
用于验证 kernel 编译成功的测试程序。

## Kernel 变体

提取的 kernel 包含两个优化变体：

### 1. `w4a16_sm90_gemm_128`
- **CTA Shape**: 128x128x128
- **Cluster Shape**: 1x1x1
- **Mainloop Schedule**: TMA Warp Specialized Cooperative
- **Epilogue Schedule**: TMA Warp Specialized Cooperative
- **适用场景**: 大规模矩阵乘法，高吞吐量

### 2. `w4a16_sm90_gemm_64`
- **CTA Shape**: 64x128x128
- **Cluster Shape**: 1x1x1
- **Mainloop Schedule**: TMA Warp Specialized Pingpong
- **Epilogue Schedule**: TMA Warp Specialized
- **适用场景**: 较小矩阵或内存受限场景

## 技术特性

### W4A16 量化
- **权重**: 4-bit 整数 (INT4)
- **激活**: 16-bit 浮点 (FP16/BF16)
- **量化类型**: Fine-grained (细粒度) with scale-only
- **分组大小**: 可配置的 group_size

### Hopper 专属特性
- **TMA (Tensor Memory Accelerator)**: 利用 Hopper 架构的高效内存操作
- **Warp Specialization**: 针对 TMA 优化的 warp 调度
- **Cluster Support**: 支持多 CTA 集群配置

## 函数签名

```cpp
extern "C" void w4a16_sm90_gemm_128(
    half const* A,                      // 激活矩阵 (FP16)
    cutlass::uint4b_t const* B,        // 权重矩阵 (INT4)
    half const* weight_scales,         // 权重缩放因子
    half const* weight_zero_points,    // 权重零点 (可选)
    half const* biases,                // 偏置 (可选)
    float const alpha,                 // 缩放因子
    half* C,                           // 输出矩阵
    int m,                             // M 维度
    int n,                             // N 维度
    int k,                             // K 维度
    int const group_size,              // 量化分组大小
    CutlassGemmConfig gemm_config,     // GEMM 配置
    char* workspace,                   // 工作空间
    size_t workspace_bytes,            // 工作空间大小
    cudaStream_t stream,               // CUDA 流
    int* occupancy                     // 占用率查询 (可选)
);
```

## 编译命令

### 配置
```bash
cd /home/qianxu/trt_llm_w4a16_hopper/build
cmake ..
```

### 编译
```bash
make -j4
```

### 清理重新编译
```bash
make clean && make -j4
```

## 目录结构

```
/home/qianxu/trt_llm_w4a16_hopper/
├── build/                          # 构建目录
│   ├── bin/
│   │   └── test_w4a16_sm90        # 测试可执行文件
│   └── lib/
│       └── libw4a16_sm90_kernel.so # 共享库
├── include/                        # 头文件
│   ├── tensorrt_llm/
│   │   ├── common/                # 通用工具
│   │   ├── cutlass_extensions/    # CUTLASS 扩展 (72 个文件)
│   │   └── kernels/               # Kernel 头文件
│   └── NvInferRuntime.h           # TensorRT stub
├── src/                           # 源文件
│   ├── w4a16_sm90_kernel.cu       # Kernel wrapper
│   ├── test_main.cu               # 测试主程序
│   ├── logger.cpp                 # 日志实现
│   ├── stringUtils.cpp            # 字符串工具
│   ├── assert.cpp                 # 断言实现
│   └── tllmException.cpp          # 异常处理
├── CMakeLists.txt                 # CMake 配置
├── build.sh                       # 构建脚本
└── README.md                      # 项目说明

```

## 依赖项

### 必需
- CUDA Toolkit 12.x+
- CMake 3.18+
- C++17 兼容编译器
- CUTLASS (从 TensorRT-LLM 获取)

### 可选
- NVIDIA GPU with SM90+ (用于运行)

## 使用示例

### 在代码中使用

```cpp
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// 声明 kernel 函数
extern "C" void w4a16_sm90_gemm_128(...);

int main() {
    // 分配内存
    half *d_A, *d_C, *d_scales;
    cutlass::uint4b_t *d_B;

    // ... 初始化数据 ...

    // 调用 kernel
    w4a16_sm90_gemm_128(
        d_A, d_B, d_scales, nullptr, nullptr,
        1.0f, d_C, m, n, k, 128,
        gemm_config, workspace, workspace_bytes,
        stream, nullptr
    );

    cudaDeviceSynchronize();

    return 0;
}
```

### 链接库

```bash
g++ your_code.cpp -o your_app \
    -L/home/qianxu/trt_llm_w4a16_hopper/build/lib \
    -lw4a16_sm90_kernel \
    -lcudart \
    -I/home/qianxu/trt_llm_w4a16_hopper/include
```

## 性能特点

### 优势
1. **Hopper 优化**: 充分利用 H100/H200 的 TMA 硬件加速
2. **内存高效**: 4-bit 权重降低 4倍 内存占用
3. **向后兼容**: SM120 (RTX 5070) 可以运行 SM90 kernels
4. **灵活配置**: 支持多种 CTA/Cluster 配置

### 适用场景
- 大语言模型推理 (LLM Inference)
- W4A16 量化模型部署
- 内存受限的高性能计算
- Hopper 架构 GPU 加速

## 注意事项

1. **架构兼容性**: 虽然编译目标是 SM90，但 SM120 (RTX 5070) 向后兼容
2. **量化精度**: W4A16 会有轻微精度损失，但通常可以接受
3. **Group Size**: 必须是 CTA K 维度的倍数 (通常是 128)
4. **工作空间**: 某些配置需要额外的 GPU 内存作为工作空间

## 验证测试

运行测试程序验证编译：
```bash
cd /home/qianxu/trt_llm_w4a16_hopper/build
./bin/test_w4a16_sm90
```

预期输出：
```
Testing W4A16 Hopper (SM90) Kernel...
=====================================

W4A16 SM90 Hopper kernel compiled successfully!
Available kernel variants:
  - w4a16_sm90_gemm_128: 128x128x128 CTA with TMA Cooperative
  - w4a16_sm90_gemm_64:  64x128x128 CTA with TMA Pingpong

Kernel library compiled and linked successfully!
Library location: lib/libw4a16_sm90_kernel.so
```

## 下一步

1. **性能测试**: 在实际硬件上进行基准测试
2. **功能验证**: 使用真实数据验证计算正确性
3. **集成**: 集成到你的推理框架或应用中
4. **优化**: 根据具体使用场景调整 CTA/Cluster 配置

## 致谢

此 kernel 提取自 [NVIDIA TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) 项目。

---

**构建状态**: ✅ 成功
**最后更新**: 2025-11-14 15:27
