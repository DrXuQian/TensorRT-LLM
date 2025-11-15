/*
 * W4A16 SM90 测试程序
 * 测试 FP16 激活 + INT4 权重的 GEMM kernel
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <random>
#include <vector>

// TensorRT-LLM includes
// Note: COMPILE_HOPPER_TMA_GEMMS is defined in build script
#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm_template.h"

using namespace tensorrt_llm::kernels::cutlass_kernels;
using namespace tensorrt_llm::cutlass_extensions;

// 显式实例化 W4A16 SM90 runner
namespace tensorrt_llm {
namespace kernels {
namespace cutlass_kernels {
template class CutlassFpAIntBGemmRunner<half, cutlass::uint4b_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY>;
}}}

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

int main(int argc, char** argv)
{
    printf("=== W4A16 SM90 Test ===\n\n");

    // 检查 GPU
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n\n", prop.major, prop.minor);

    if (prop.major != 9 || prop.minor != 0) {
        fprintf(stderr, "此程序需要 H100/H800 GPU (Compute Capability 9.0)\n");
        fprintf(stderr, "当前 GPU: CC %d.%d\n", prop.major, prop.minor);
        return 1;
    }

    // 矩阵维度
    int M = (argc > 1) ? atoi(argv[1]) : 512;
    int N = (argc > 2) ? atoi(argv[2]) : 1024;
    int K = (argc > 3) ? atoi(argv[3]) : 1024;
    int group_size = 128;

    printf("Matrix: M=%d, N=%d, K=%d, group_size=%d\n\n", M, N, K, group_size);

    // 分配内存
    size_t size_A = M * K;
    size_t size_B = N * K / 2;
    size_t size_C = M * N;
    size_t size_scales = N * (K / group_size);

    half *d_A, *d_C, *d_scales;
    uint8_t *d_B;

    printf("Allocating memory...\n");
    CUDA_CHECK(cudaMalloc(&d_A, size_A * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_scales, size_scales * sizeof(half)));

    // 初始化数据
    printf("Initializing data...\n");
    std::vector<half> h_A(size_A);
    std::vector<uint8_t> h_B(size_B);
    std::vector<half> h_scales(size_scales);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    std::uniform_int_distribution<int> dis_int(0, 15);

    for (size_t i = 0; i < size_A; i++) {
        h_A[i] = __float2half(dis(gen));
    }

    for (size_t i = 0; i < size_B; i++) {
        uint8_t val1 = dis_int(gen);
        uint8_t val2 = dis_int(gen);
        h_B[i] = (val2 << 4) | val1;
    }

    for (size_t i = 0; i < size_scales; i++) {
        h_scales[i] = __float2half(0.1f);
    }

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scales, h_scales.data(), size_scales * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, size_C * sizeof(half)));

    printf("\n=== Creating CutlassFpAIntBGemmRunner ===\n");

    CutlassFpAIntBGemmRunner<half, cutlass::uint4b_t,
        cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> runner;

    printf("Runner created!\n");

    // 获取 workspace
    size_t workspace_bytes = runner.getWorkspaceSize(M, N, K);
    printf("Workspace: %.2f MB\n", workspace_bytes / (1024.0 * 1024.0));

    char* d_workspace = nullptr;
    if (workspace_bytes > 0) {
        CUDA_CHECK(cudaMalloc(&d_workspace, workspace_bytes));
    }

    // 获取配置
    auto configs = runner.getConfigs();
    printf("Available configs: %zu\n", configs.size());

    CutlassGemmConfig gemm_config;
    if (!configs.empty()) {
        gemm_config = configs[0];
        printf("Using first config\n");
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    printf("\n=== Running GEMM ===\n");

    try {
        runner.gemm(
            d_A,
            reinterpret_cast<cutlass::uint4b_t*>(d_B),
            d_scales,
            nullptr,  // zero points
            nullptr,  // biases
            1.0f,     // alpha
            d_C,
            M, N, K,
            group_size,
            gemm_config,
            d_workspace,
            workspace_bytes,
            stream
        );

        printf("GEMM call returned\n");

    } catch (const std::exception& e) {
        fprintf(stderr, "Exception: %s\n", e.what());
        return 1;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Synchronizing...\n");
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "Sync error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("✅ GEMM completed!\n\n");

    // 检查输出是否非零
    printf("=== Checking Output ===\n");
    std::vector<half> h_C(size_C);
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size_C * sizeof(half), cudaMemcpyDeviceToHost));

    int non_zero_count = 0;
    float sum = 0.0f;
    for (size_t i = 0; i < size_C; i++) {
        float val = __half2float(h_C[i]);
        if (val != 0.0f) {
            non_zero_count++;
            sum += val;
        }
    }

    printf("Non-zero elements: %d / %zu (%.2f%%)\n",
           non_zero_count, size_C, (non_zero_count * 100.0f) / size_C);
    printf("Average value: %.6f\n", sum / size_C);

    // 显示前几个值
    printf("\nFirst 10 output values:\n");
    for (int i = 0; i < std::min(10, (int)size_C); i++) {
        printf("  C[%d] = %.6f\n", i, __half2float(h_C[i]));
    }

    bool passed = (non_zero_count > 0);

    printf("\n=== Test Result ===\n");
    if (passed) {
        printf("✅ PASSED: Kernel executed successfully with non-zero output\n");
    } else {
        printf("❌ FAILED: All outputs are zero\n");
    }

    // 清理
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_scales));
    if (d_workspace) CUDA_CHECK(cudaFree(d_workspace));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return passed ? 0 : 1;
}
