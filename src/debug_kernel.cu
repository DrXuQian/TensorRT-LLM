/*
 * Debug version - 检查为什么输出全为0
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cutlass/numeric_types.h>
#include <stdio.h>
#include <stdlib.h>

// 声明 kernel 函数
extern "C" void w4a16_sm90_gemm_128(
    half const* A,
    cutlass::uint4b_t const* B,
    half const* weight_scales,
    half const* weight_zero_points,
    half const* biases,
    float const alpha,
    half* C,
    int m, int n, int k,
    int const group_size,
    void* gemm_config,
    char* workspace,
    size_t workspace_bytes,
    cudaStream_t stream,
    int* occupancy
);

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

int main() {
    printf("=== W4A16 Kernel Debug Test ===\n\n");

    // 检查 GPU
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n\n", prop.major, prop.minor);

    // 小矩阵测试
    int M = 32, N = 64, K = 128;
    int group_size = 128;

    printf("Test Config: M=%d, N=%d, K=%d, group_size=%d\n\n", M, N, K, group_size);

    // 分配 host 内存
    size_t size_A = M * K;
    size_t size_B = N * K / 2; // INT4
    size_t size_C = M * N;
    size_t size_scales = N * (K / group_size);

    half* h_A = (half*)malloc(size_A * sizeof(half));
    uint8_t* h_B = (uint8_t*)malloc(size_B);
    half* h_C = (half*)malloc(size_C * sizeof(half));
    half* h_scales = (half*)malloc(size_scales * sizeof(half));

    // 初始化为简单值
    printf("Initializing data...\n");
    for (size_t i = 0; i < size_A; i++) {
        h_A[i] = __float2half(1.0f); // 全1
    }
    for (size_t i = 0; i < size_B; i++) {
        h_B[i] = 0x11; // INT4: 每个元素为1
    }
    for (size_t i = 0; i < size_scales; i++) {
        h_scales[i] = __float2half(1.0f); // scale = 1
    }
    for (size_t i = 0; i < size_C; i++) {
        h_C[i] = __float2half(0.0f);
    }

    // 打印输入样本
    printf("Input A[0-5]: ");
    for (int i = 0; i < 5; i++) {
        printf("%.2f ", __half2float(h_A[i]));
    }
    printf("\n");

    printf("Input B[0-5]: ");
    for (int i = 0; i < 5; i++) {
        printf("0x%02x ", h_B[i]);
    }
    printf("\n");

    printf("Scales[0-5]: ");
    for (int i = 0; i < std::min(5, (int)size_scales); i++) {
        printf("%.2f ", __half2float(h_scales[i]));
    }
    printf("\n\n");

    // 分配 device 内存
    half *d_A, *d_C, *d_scales;
    uint8_t* d_B;
    char* d_workspace;

    CUDA_CHECK(cudaMalloc(&d_A, size_A * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_scales, size_scales * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_workspace, 8*1024*1024));

    // 拷贝到 device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scales, h_scales, size_scales * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, size_C * sizeof(half)));

    printf("Running kernel...\n");

    // 调用 kernel
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    w4a16_sm90_gemm_128(
        d_A,
        reinterpret_cast<cutlass::uint4b_t const*>(d_B),
        d_scales,
        nullptr,  // no zero points
        nullptr,  // no bias
        1.0f,     // alpha
        d_C,
        M, N, K,
        group_size,
        nullptr,  // default config
        d_workspace,
        8*1024*1024,
        stream,
        nullptr
    );

    // 检查 kernel 错误
    cudaError_t kernel_err = cudaGetLastError();
    if (kernel_err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(kernel_err));
        return 1;
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    printf("Kernel completed successfully!\n\n");

    // 拷贝结果
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C * sizeof(half), cudaMemcpyDeviceToHost));

    // 检查输出
    printf("Output C[0-20]:\n");
    int zero_count = 0;
    int nonzero_count = 0;

    for (int i = 0; i < std::min(20, M * N); i++) {
        float val = __half2float(h_C[i]);
        printf("  C[%d] = %.6f\n", i, val);
        if (val == 0.0f) {
            zero_count++;
        } else {
            nonzero_count++;
        }
    }

    printf("\n=== Statistics ===\n");
    printf("Total elements: %d\n", M * N);
    printf("Non-zero count (first 20): %d\n", nonzero_count);
    printf("Zero count (first 20): %d\n", zero_count);

    // 检查全部输出
    zero_count = 0;
    nonzero_count = 0;
    float sum = 0.0f;
    float max_val = -1e9f;
    float min_val = 1e9f;

    for (int i = 0; i < M * N; i++) {
        float val = __half2float(h_C[i]);
        sum += val;
        if (val == 0.0f) zero_count++;
        else nonzero_count++;
        if (val > max_val) max_val = val;
        if (val < min_val) min_val = val;
    }

    printf("All elements - Zero: %d, Non-zero: %d\n", zero_count, nonzero_count);
    printf("Sum: %.6f\n", sum);
    printf("Mean: %.6f\n", sum / (M * N));
    printf("Max: %.6f\n", max_val);
    printf("Min: %.6f\n", min_val);

    if (zero_count == M * N) {
        printf("\n❌ ERROR: All outputs are zero!\n");
        printf("Possible causes:\n");
        printf("  1. Kernel not executing properly\n");
        printf("  2. Input data format issue\n");
        printf("  3. Configuration issue (gemm_config)\n");
        printf("  4. INT4 packing issue\n");
    } else {
        printf("\n✅ SUCCESS: Kernel produced non-zero output!\n");
    }

    // 清理
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_scales);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_scales));
    CUDA_CHECK(cudaFree(d_workspace));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return 0;
}
