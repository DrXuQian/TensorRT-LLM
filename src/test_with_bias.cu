/*
 * 测试版本 - 提供真实的 bias 而不是 nullptr
 * 因为 TMA 可能不能处理 nullptr
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cutlass/numeric_types.h>
#include <stdio.h>

extern "C" void w4a16_sm90_gemm_128(
    half const* A, cutlass::uint4b_t const* B,
    half const* weight_scales, half const* weight_zero_points,
    half const* biases, float const alpha, half* C,
    int m, int n, int k, int const group_size,
    void* gemm_config, char* workspace, size_t workspace_bytes,
    cudaStream_t stream, int* occupancy
);

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

int main(int argc, char** argv) {
    printf("=== W4A16 Test WITH Real Bias ===\n\n");

    int M = (argc > 1) ? atoi(argv[1]) : 1024;
    int N = (argc > 2) ? atoi(argv[2]) : 2048;
    int K = (argc > 3) ? atoi(argv[3]) : 2048;
    int group_size = 128;

    printf("Config: M=%d, N=%d, K=%d, group_size=%d\n\n", M, N, K, group_size);

    // 计算大小
    size_t size_A = M * K;
    size_t size_B = N * K / 2;
    size_t size_C = M * N;
    size_t size_scales = N * (K / group_size);
    size_t size_bias = N;  // bias 是 N 维向量

    printf("Allocating memory...\n");
    half *d_A, *d_C, *d_scales, *d_bias;
    uint8_t *d_B;
    char *d_workspace;

    CUDA_CHECK(cudaMalloc(&d_A, size_A * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_scales, size_scales * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_bias, size_bias * sizeof(half)));  // ✅ 分配真实的 bias
    CUDA_CHECK(cudaMalloc(&d_workspace, 16*1024*1024));

    printf("Initializing data...\n");
    // 初始化为有意义的值
    CUDA_CHECK(cudaMemset(d_A, 0x3C, size_A * sizeof(half)));      // ~1.0
    CUDA_CHECK(cudaMemset(d_B, 0x11, size_B));                      // INT4: 1
    CUDA_CHECK(cudaMemset(d_scales, 0x3C, size_scales * sizeof(half))); // ~1.0
    CUDA_CHECK(cudaMemset(d_bias, 0x38, size_bias * sizeof(half)));    // ~0.5
    CUDA_CHECK(cudaMemset(d_C, 0, size_C * sizeof(half)));

    printf("Launching kernel WITH BIAS...\n");
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // ✅ 传入真实的 bias 而不是 nullptr
    w4a16_sm90_gemm_128(
        d_A,
        reinterpret_cast<cutlass::uint4b_t const*>(d_B),
        d_scales,
        nullptr,    // no zero points
        d_bias,     // ✅ 传入真实 bias
        1.0f,
        d_C,
        M, N, K,
        group_size,
        nullptr,
        d_workspace,
        16*1024*1024,
        stream,
        nullptr
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "❌ Kernel error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    printf("✅ Kernel completed!\n\n");

    // 检查输出
    half* h_C = (half*)malloc(std::min((size_t)100, size_C) * sizeof(half));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, std::min((size_t)100, size_C) * sizeof(half), cudaMemcpyDeviceToHost));

    printf("First 10 outputs:\n");
    int zero_count = 0, nonzero_count = 0;
    for (int i = 0; i < std::min(10, (int)size_C); i++) {
        float val = __half2float(h_C[i]);
        printf("  C[%d] = %.6f\n", i, val);
        if (val == 0.0f) zero_count++;
        else nonzero_count++;
    }

    for (int i = 10; i < std::min(100, (int)size_C); i++) {
        float val = __half2float(h_C[i]);
        if (val == 0.0f) zero_count++;
        else nonzero_count++;
    }

    printf("\nStatistics (first 100):\n");
    printf("  Zero: %d\n", zero_count);
    printf("  Non-zero: %d\n", nonzero_count);

    if (nonzero_count > 0) {
        printf("\n✅ SUCCESS: Output has non-zero values!\n");
    } else {
        printf("\n❌ FAILURE: All outputs are still zero!\n");
    }

    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_scales));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_workspace));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return 0;
}
