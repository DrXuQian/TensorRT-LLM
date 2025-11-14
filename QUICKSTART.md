# W4A16 Hopper Kernel - 快速开始指南

## 编译完成 ✓

W4A16 Hopper (SM90) kernel 已经成功提取并编译！

## 文件位置

### 库文件
```
/home/qianxu/trt_llm_w4a16_hopper/build/lib/libw4a16_sm90_kernel.so (2.7 MB)
```

### 测试程序
```
/home/qianxu/trt_llm_w4a16_hopper/build/bin/test_w4a16_sm90 (19 KB)
```

## 快速测试

```bash
cd /home/qianxu/trt_llm_w4a16_hopper/build
./bin/test_w4a16_sm90
```

## 重新编译

如果需要重新编译：

```bash
cd /home/qianxu/trt_llm_w4a16_hopper/build
make clean
make -j4
```

或者完全重新配置：

```bash
cd /home/qianxu/trt_llm_w4a16_hopper
rm -rf build
mkdir build
cd build
cmake ..
make -j4
```

## Kernel API

提供了两个优化的 kernel 函数：

### 1. w4a16_sm90_gemm_128 (大矩阵优化)

```cpp
extern "C" void w4a16_sm90_gemm_128(
    half const* A,                   // 输入激活 [M, K]
    cutlass::uint4b_t const* B,     // INT4 权重 [N, K]
    half const* weight_scales,       // 权重缩放因子 [N, K/group_size]
    half const* weight_zero_points,  // 零点 (可为 nullptr)
    half const* biases,             // 偏置 (可为 nullptr)
    float const alpha,              // 缩放因子
    half* C,                        // 输出 [M, N]
    int m, int n, int k,            // 矩阵维度
    int const group_size,           // 量化分组大小 (通常 128)
    CutlassGemmConfig gemm_config,  // GEMM 配置
    char* workspace,                // 工作空间
    size_t workspace_bytes,         // 工作空间大小
    cudaStream_t stream,            // CUDA 流
    int* occupancy                  // 占用率查询 (可为 nullptr)
);
```

### 2. w4a16_sm90_gemm_64 (小矩阵/内存优化)

相同的函数签名，但使用不同的 CTA 配置。

## 在你的项目中使用

### 方法 1: 链接共享库

```bash
g++ -o my_app my_app.cpp \
    -L/home/qianxu/trt_llm_w4a16_hopper/build/lib \
    -lw4a16_sm90_kernel \
    -lcudart \
    -I/home/qianxu/trt_llm_w4a16_hopper/include \
    -I/home/qianxu/TensorRT-LLM/3rdparty/cutlass/include
```

### 方法 2: 运行时加载

```cpp
#include <dlfcn.h>

void* handle = dlopen("libw4a16_sm90_kernel.so", RTLD_LAZY);
auto kernel = (decltype(&w4a16_sm90_gemm_128))dlsym(handle, "w4a16_sm90_gemm_128");
// 使用 kernel...
dlclose(handle);
```

## 简单示例

```cpp
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>

extern "C" void w4a16_sm90_gemm_128(
    half const* A, void const* B,
    half const* scales, half const* zeros,
    half const* bias, float alpha, half* C,
    int m, int n, int k, int group_size,
    void* config, char* workspace, size_t ws_bytes,
    cudaStream_t stream, int* occ
);

int main() {
    // 矩阵维度
    int M = 1024, N = 4096, K = 4096;
    int group_size = 128;

    // 分配设备内存
    half *d_A, *d_C, *d_scales;
    void *d_B;
    char *d_workspace;

    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, N * K / 2);  // INT4 每个元素 0.5 字节
    cudaMalloc(&d_scales, N * (K/group_size) * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(half));
    cudaMalloc(&d_workspace, 4*1024*1024);  // 4MB 工作空间

    // 初始化数据...
    // (你的数据加载代码)

    // 调用 kernel
    w4a16_sm90_gemm_128(
        d_A, d_B, d_scales,
        nullptr,  // 无零点
        nullptr,  // 无偏置
        1.0f,     // alpha
        d_C,
        M, N, K,
        group_size,
        nullptr,  // 默认配置
        d_workspace,
        4*1024*1024,
        0,        // 默认流
        nullptr   // 不查询占用率
    );

    cudaDeviceSynchronize();

    // 释放内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_scales);
    cudaFree(d_C);
    cudaFree(d_workspace);

    return 0;
}
```

编译示例：
```bash
nvcc example.cpp -o example \
    -L/home/qianxu/trt_llm_w4a16_hopper/build/lib \
    -lw4a16_sm90_kernel \
    -I/home/qianxu/trt_llm_w4a16_hopper/include
```

## 兼容性

- **目标架构**: SM90 (Hopper)
- **测试架构**: SM120 (RTX 5070) - 向后兼容 ✓
- **CUDA 版本**: 12.8+ 推荐
- **量化格式**: W4A16 (4-bit weights, 16-bit activations)

## 性能提示

1. **选择正确的 kernel**:
   - 大矩阵 (M×N×K > 1M): 使用 `w4a16_sm90_gemm_128`
   - 小矩阵或内存受限: 使用 `w4a16_sm90_gemm_64`

2. **Group Size**:
   - 推荐值: 128 或 64
   - 必须是 128 的倍数（对于 128x128x128 CTA）

3. **工作空间**:
   - 分配足够的工作空间（建议 4-8 MB）
   - 可以复用工作空间以节省内存

4. **CUDA 流**:
   - 使用不同的流来并行执行多个 kernel

## 故障排除

### 编译错误

如果遇到编译错误：
```bash
cd /home/qianxu/trt_llm_w4a16_hopper/build
make clean
cmake .. -DCUTLASS_DIR=/home/qianxu/TensorRT-LLM/3rdparty/cutlass
make -j4
```

### 运行时错误

- 检查 GPU 架构是否兼容（需要 SM90+）
- 确保分配了足够的工作空间
- 验证 group_size 是否正确（必须整除 K）

### 链接错误

确保库路径正确：
```bash
export LD_LIBRARY_PATH=/home/qianxu/trt_llm_w4a16_hopper/build/lib:$LD_LIBRARY_PATH
```

## 更多信息

- 详细文档: [BUILD_SUCCESS.md](BUILD_SUCCESS.md)
- 提取摘要: [EXTRACTION_SUMMARY.md](EXTRACTION_SUMMARY.md)
- 完整说明: [README.md](README.md)

## Git 历史

查看提取和构建历史：
```bash
git log --oneline
```

每个 commit 都对应提取过程的一个步骤。

---

🎉 恭喜！你已经成功提取并编译了 W4A16 Hopper kernel！
