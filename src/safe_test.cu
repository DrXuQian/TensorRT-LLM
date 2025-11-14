/*
 * 安全测试 - 尝试捕获异常
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cutlass/numeric_types.h>
#include <stdio.h>
#include <exception>
#include <stdexcept>

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

int main() {
    printf("=== Safe Test with Exception Handling ===\n\n");

    int M = 1024, N = 2048, K = 2048;
    int group_size = 128;

    printf("Config: M=%d, N=%d, K=%d\n\n", M, N, K);

    size_t size_A = M * K;
    size_t size_B = N * K / 2;
    size_t size_C = M * N;
    size_t size_scales = N * (K / group_size);
    size_t size_bias = N;

    half *d_A, *d_C, *d_scales, *d_bias;
    uint8_t *d_B;
    char *d_workspace;

    printf("Allocating memory...\n");
    CUDA_CHECK(cudaMalloc(&d_A, size_A * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_scales, size_scales * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_bias, size_bias * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_workspace, 16*1024*1024));

    printf("Initializing data...\n");
    CUDA_CHECK(cudaMemset(d_A, 0x3C, size_A * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_B, 0x11, size_B));
    CUDA_CHECK(cudaMemset(d_scales, 0x3C, size_scales * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_bias, 0x38, size_bias * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_C, 0, size_C * sizeof(half)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    printf("\nCalling kernel inside try-catch block...\n");

    try {
        printf("  Entering try block...\n");

        w4a16_sm90_gemm_128(
            d_A,
            reinterpret_cast<cutlass::uint4b_t const*>(d_B),
            d_scales,
            nullptr,  // no zero points
            d_bias,   // with bias
            1.0f,
            d_C,
            M, N, K,
            group_size,
            nullptr,  // default config
            d_workspace,
            16*1024*1024,
            stream,
            nullptr
        );

        printf("  Kernel call returned successfully!\n");

    } catch (const std::exception& e) {
        fprintf(stderr, "\n❌ Caught exception: %s\n", e.what());
        return 1;
    } catch (...) {
        fprintf(stderr, "\n❌ Caught unknown exception\n");
        return 1;
    }

    printf("  Checking for kernel errors...\n");
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "❌ Kernel error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("  ✅ No kernel launch errors\n");

    printf("  Synchronizing stream...\n");
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "❌ Sync error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("  ✅ Stream synchronized\n\n");

    printf("✅ KERNEL EXECUTED SUCCESSFULLY!\n\n");

    // 检查输出
    half* h_C = (half*)malloc(10 * sizeof(half));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, 10 * sizeof(half), cudaMemcpyDeviceToHost));

    printf("First 10 outputs:\n");
    for (int i = 0; i < 10; i++) {
        printf("  C[%d] = %.6f\n", i, __half2float(h_C[i]));
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
