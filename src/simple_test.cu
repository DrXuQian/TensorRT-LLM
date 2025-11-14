/*
 * 最简单的测试 - 检查 kernel 是否真的被启动
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
        fprintf(stderr, "❌ CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } else { \
        printf("✅ %s succeeded\n", #call); \
    } \
} while(0)

int main(int argc, char** argv) {
    printf("=== Simple W4A16 Kernel Test ===\n\n");

    // 从命令行读取矩阵尺寸，默认使用较大的尺寸
    int M = (argc > 1) ? atoi(argv[1]) : 2048;
    int N = (argc > 2) ? atoi(argv[2]) : 4096;
    int K = (argc > 3) ? atoi(argv[3]) : 4096;
    int group_size = 128;

    printf("Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    printf("Group size: %d\n\n", group_size);

    // 检查设备
    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("GPU: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    // 计算大小
    size_t size_A = M * K;
    size_t size_B = N * K / 2; // INT4
    size_t size_C = M * N;
    size_t size_scales = N * (K / group_size);

    printf("Allocating memory...\n");
    printf("  A: %.2f MB\n", size_A * sizeof(half) / 1024.0 / 1024.0);
    printf("  B: %.2f MB\n", size_B / 1024.0 / 1024.0);
    printf("  C: %.2f MB\n", size_C * sizeof(half) / 1024.0 / 1024.0);
    printf("  Scales: %.2f MB\n\n", size_scales * sizeof(half) / 1024.0 / 1024.0);

    // 分配设备内存
    half *d_A, *d_C, *d_scales;
    uint8_t *d_B;
    char *d_workspace;

    CUDA_CHECK(cudaMalloc(&d_A, size_A * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_scales, size_scales * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_workspace, 16*1024*1024)); // 16MB

    // 初始化为非零值
    printf("Initializing data with values...\n");
    CUDA_CHECK(cudaMemset(d_A, 0x3C, size_A * sizeof(half))); // ~1.0 in FP16
    CUDA_CHECK(cudaMemset(d_B, 0x11, size_B)); // INT4: 1,1,1,1...
    CUDA_CHECK(cudaMemset(d_scales, 0x3C, size_scales * sizeof(half))); // ~1.0
    CUDA_CHECK(cudaMemset(d_C, 0, size_C * sizeof(half))); // Zero output

    printf("\n🚀 Launching kernel...\n");
    printf("   Function: w4a16_sm90_gemm_128\n");
    printf("   Alpha: 1.0\n");
    printf("   Bias: nullptr\n");
    printf("   Zero points: nullptr\n\n");

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // 调用 kernel
    w4a16_sm90_gemm_128(
        d_A,
        reinterpret_cast<cutlass::uint4b_t const*>(d_B),
        d_scales,
        nullptr, // no zero points
        nullptr, // no bias
        1.0f,
        d_C,
        M, N, K,
        group_size,
        nullptr, // default config
        d_workspace,
        16*1024*1024,
        stream,
        nullptr
    );

    // 检查 kernel 错误
    cudaError_t kernel_err = cudaGetLastError();
    if (kernel_err != cudaSuccess) {
        fprintf(stderr, "❌ Kernel launch failed: %s\n", cudaGetErrorString(kernel_err));
        return 1;
    }
    printf("✅ Kernel launched without errors\n");

    // 同步
    printf("Waiting for kernel to complete...\n");
    cudaError_t sync_err = cudaStreamSynchronize(stream);
    if (sync_err != cudaSuccess) {
        fprintf(stderr, "❌ Kernel execution failed: %s\n", cudaGetErrorString(sync_err));
        return 1;
    }
    printf("✅ Kernel completed successfully\n\n");

    // 检查输出
    printf("Checking output...\n");
    half* h_C = (half*)malloc(std::min((size_t)100, size_C) * sizeof(half));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, std::min((size_t)100, size_C) * sizeof(half), cudaMemcpyDeviceToHost));

    int zero_count = 0;
    int nonzero_count = 0;
    float sum = 0.0f;

    printf("First 10 output values:\n");
    for (int i = 0; i < std::min(10, (int)size_C); i++) {
        float val = __half2float(h_C[i]);
        printf("  C[%d] = %.6f\n", i, val);
        if (val == 0.0f) zero_count++;
        else {
            nonzero_count++;
            sum += val;
        }
    }

    // 统计前100个
    for (int i = 10; i < std::min(100, (int)size_C); i++) {
        float val = __half2float(h_C[i]);
        if (val == 0.0f) zero_count++;
        else {
            nonzero_count++;
            sum += val;
        }
    }

    printf("\n=== Results (first 100 elements) ===\n");
    printf("Zero count: %d\n", zero_count);
    printf("Non-zero count: %d\n", nonzero_count);
    if (nonzero_count > 0) {
        printf("Sum: %.6f\n", sum);
        printf("Average: %.6f\n", sum / nonzero_count);
    }

    if (zero_count == std::min(100, (int)size_C)) {
        printf("\n❌ WARNING: All sampled outputs are ZERO!\n");
        printf("This suggests the kernel may not have executed properly.\n");
        printf("\nPossible causes:\n");
        printf("1. Kernel was filtered out due to problem size\n");
        printf("2. Input data format issue (INT4 packing)\n");
        printf("3. Missing bias causes zero output\n");
        printf("4. Configuration issue\n");
    } else {
        printf("\n✅ SUCCESS: Kernel produced non-zero outputs!\n");
    }

    // 清理
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_scales));
    CUDA_CHECK(cudaFree(d_workspace));
    CUDA_CHECK(cudaStreamDestroy(stream));

    printf("\n✅ Test completed\n");
    return 0;
}
