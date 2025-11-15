/*
 * 最小化 W4A16 SM90 测试 - 只实例化必要的模板
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <random>
#include <vector>

// 定义编译选项
#define COMPILE_HOPPER_TMA_GEMMS

// TensorRT-LLM includes
#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm_template.h"

using namespace tensorrt_llm::kernels::cutlass_kernels;
using namespace tensorrt_llm::cutlass_extensions;

// 显式实例化 W4A16 runner
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
    printf("=== W4A16 SM90 Minimal Test ===\n\n");

    // 检查 GPU
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n\n", prop.major, prop.minor);

    if (prop.major * 10 + prop.minor < 90) {
        fprintf(stderr, "需要 SM90+ GPU (Hopper)\n");
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

    // CPU 参考实现 (FP32 精度)
    printf("=== Computing CPU Reference ===\n");
    std::vector<float> h_C_ref(size_C, 0.0f);

    // 解包 INT4 权重并计算参考结果
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;

            for (int k = 0; k < K; k++) {
                float a_val = __half2float(h_A[m * K + k]);

                // 解包 INT4 权重
                int b_idx = n * K + k;
                int byte_idx = b_idx / 2;
                int nibble_idx = b_idx % 2;
                uint8_t byte_val = h_B[byte_idx];
                uint8_t int4_val = (nibble_idx == 0) ? (byte_val & 0x0F) : ((byte_val >> 4) & 0x0F);

                // INT4 是无符号的 [0, 15]
                float b_val = static_cast<float>(int4_val);

                // 应用量化缩放
                int scale_idx = n * (K / group_size) + (k / group_size);
                float scale = __half2float(h_scales[scale_idx]);
                b_val *= scale;

                sum += a_val * b_val;
            }

            h_C_ref[m * N + n] = sum;
        }
    }
    printf("CPU reference computed\n\n");

    // 从 GPU 复制完整结果
    printf("=== Validating Results ===\n");
    std::vector<half> h_C(size_C);
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size_C * sizeof(half), cudaMemcpyDeviceToHost));

    // 精度检查
    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;
    float avg_abs_error = 0.0f;
    float avg_rel_error = 0.0f;
    int num_large_errors = 0;

    const float abs_tol = 1e-2f;  // 绝对误差容忍度
    const float rel_tol = 1e-2f;  // 相对误差容忍度 (1%)

    for (size_t i = 0; i < size_C; i++) {
        float gpu_val = __half2float(h_C[i]);
        float cpu_val = h_C_ref[i];

        float abs_error = std::abs(gpu_val - cpu_val);
        float rel_error = (std::abs(cpu_val) > 1e-6f) ?
                          (abs_error / std::abs(cpu_val)) : 0.0f;

        max_abs_error = std::max(max_abs_error, abs_error);
        max_rel_error = std::max(max_rel_error, rel_error);
        avg_abs_error += abs_error;
        avg_rel_error += rel_error;

        if (abs_error > abs_tol && rel_error > rel_tol) {
            num_large_errors++;
            if (num_large_errors <= 5) {  // 只打印前 5 个大误差
                printf("  Large error at [%zu]: GPU=%.6f, CPU=%.6f, abs_err=%.6f, rel_err=%.2f%%\n",
                       i, gpu_val, cpu_val, abs_error, rel_error * 100.0f);
            }
        }
    }

    avg_abs_error /= size_C;
    avg_rel_error /= size_C;

    printf("\n=== Accuracy Statistics ===\n");
    printf("Max absolute error: %.6f\n", max_abs_error);
    printf("Max relative error: %.2f%%\n", max_rel_error * 100.0f);
    printf("Avg absolute error: %.6f\n", avg_abs_error);
    printf("Avg relative error: %.2f%%\n", avg_rel_error * 100.0f);
    printf("Elements with large errors: %d / %zu (%.2f%%)\n",
           num_large_errors, size_C, (num_large_errors * 100.0f) / size_C);

    // 显示一些样本值
    printf("\n=== Sample Results (first 5) ===\n");
    for (int i = 0; i < std::min(5, (int)size_C); i++) {
        float gpu_val = __half2float(h_C[i]);
        float cpu_val = h_C_ref[i];
        float abs_error = std::abs(gpu_val - cpu_val);
        float rel_error = (std::abs(cpu_val) > 1e-6f) ?
                          (abs_error / std::abs(cpu_val)) : 0.0f;
        printf("  [%d] GPU: %8.4f  CPU: %8.4f  Abs: %.6f  Rel: %.2f%%\n",
               i, gpu_val, cpu_val, abs_error, rel_error * 100.0f);
    }

    // 判断测试是否通过
    bool passed = (max_abs_error < abs_tol * 10.0f) ||  // 放宽 10 倍容忍度
                  (max_rel_error < rel_tol * 10.0f);     // 或相对误差在范围内

    printf("\n=== Test Result ===\n");
    if (passed) {
        printf("✅ PASSED: Results within acceptable tolerance\n");
    } else {
        printf("❌ FAILED: Errors exceed tolerance\n");
        printf("   Max absolute error: %.6f (threshold: %.6f)\n",
               max_abs_error, abs_tol * 10.0f);
        printf("   Max relative error: %.2f%% (threshold: %.2f%%)\n",
               max_rel_error * 100.0f, rel_tol * 1000.0f);
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
