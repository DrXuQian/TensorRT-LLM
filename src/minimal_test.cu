/*
 * 最小测试 - 不调用 kernel，只测试基础设施
 */

#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    printf("=== Minimal CUDA Test ===\n\n");

    // 测试1: CUDA 初始化
    printf("Test 1: CUDA initialization...\n");
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        fprintf(stderr, "❌ cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("✅ Found %d CUDA device(s)\n\n", device_count);

    // 测试2: 获取设备属性
    printf("Test 2: Get device properties...\n");
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "❌ cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("✅ GPU: %s\n", prop.name);
    printf("✅ Compute Capability: %d.%d\n\n", prop.major, prop.minor);

    // 测试3: 内存分配
    printf("Test 3: Memory allocation...\n");
    void* d_ptr;
    err = cudaMalloc(&d_ptr, 1024);
    if (err != cudaSuccess) {
        fprintf(stderr, "❌ cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("✅ cudaMalloc succeeded\n");

    err = cudaFree(d_ptr);
    if (err != cudaSuccess) {
        fprintf(stderr, "❌ cudaFree failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("✅ cudaFree succeeded\n\n");

    // 测试4: 创建流
    printf("Test 4: Create CUDA stream...\n");
    cudaStream_t stream;
    err = cudaStreamCreate(&stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "❌ cudaStreamCreate failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("✅ cudaStreamCreate succeeded\n");

    err = cudaStreamDestroy(stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "❌ cudaStreamDestroy failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("✅ cudaStreamDestroy succeeded\n\n");

    printf("=== All basic tests passed! ===\n");
    printf("\nNow testing if we can even link to the kernel library...\n");

    // 测试5: 尝试获取 kernel 函数的地址（不调用）
    printf("Test 5: Check if kernel library loads...\n");

    // 注意：我们只是声明，不调用
    extern "C" void w4a16_sm90_gemm_128(void*, void*, void*, void*, void*,
                                         float, void*, int, int, int, int,
                                         void*, void*, size_t, void*, void*);

    // 获取函数地址
    void* func_ptr = (void*)&w4a16_sm90_gemm_128;
    if (func_ptr == nullptr) {
        fprintf(stderr, "❌ Cannot get kernel function address\n");
        return 1;
    }
    printf("✅ Kernel function address: %p\n", func_ptr);
    printf("✅ Kernel library loaded successfully\n\n");

    printf("=== All tests passed! ===\n");
    printf("\nIf you see this, the problem is NOT with CUDA setup.\n");
    printf("The segfault must be happening INSIDE the kernel launcher.\n");

    return 0;
}
