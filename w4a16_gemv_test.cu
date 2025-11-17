/*
 * W4A16 完整测试: 结合 CUTLASS 和 CUDA Core GEMV
 * 根据 batch size 自动选择最优 kernel
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <random>
#include <vector>
#include <algorithm>
#include <cfloat>

// TensorRT-LLM includes
#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm_template.h"
#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/kernelLauncher.h"
#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/common.h"

using namespace tensorrt_llm::kernels::cutlass_kernels;
using namespace tensorrt_llm::cutlass_extensions;
namespace wo = tensorrt_llm::kernels::weight_only;

// 显式实例化 W4A16 SM90 runner
namespace tensorrt_llm {
namespace kernels {
namespace cutlass_kernels {
template class CutlassFpAIntBGemmRunner<half, cutlass::uint4b_t,
    cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY>;
}}}

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// 选择最优 CUTLASS 配置
CutlassGemmConfig selectBestConfig(
    const std::vector<CutlassGemmConfig>& configs,
    int M, int N, int K)
{
    if (configs.empty()) {
        fprintf(stderr, "No configs available!\n");
        exit(1);
    }

    int preferred_m_tile = (M >= 256) ? 128 : (M >= 128) ? 128 : 64;
    int best_idx = 0;
    int best_score = -1;

    for (size_t i = 0; i < configs.size(); i++) {
        const auto& cfg = configs[i];

        // 提取 tile shape (M, N, K) 从 SM90 config
        auto [tile_m, tile_n, tile_k] = enum_to_shape_tuple(cfg.tile_config_sm90);

        if (tile_m == 0) continue;

        int score = 0;
        // M tile 匹配度
        if (tile_m == preferred_m_tile) score += 100;
        else if (tile_m < preferred_m_tile) score += 50;

        // N tile 越大越好
        score += tile_n / 32;

        // K tile 偏好 (SM90 通常用 K=128B, 对应约 64 INT4 elements)
        if (tile_k >= 64) score += 10;

        // Cluster shape (大矩阵优先)
        if (M >= 128 && N >= 128) {
            auto [cluster_m, cluster_n, cluster_k] = enum_to_shape_tuple(cfg.cluster_shape);
            score += (cluster_m + cluster_n) * 5;
        }

        // Schedule 偏好
        if (cfg.mainloop_schedule == MainloopScheduleType::COOPERATIVE) {
            score += 20;
        }

        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }

    return configs[best_idx];
}

// 统一的 GEMM 接口
void run_w4a16_gemm(
    half* d_A,           // [M, K]
    uint8_t* d_B,        // [K, N] packed INT4
    half* d_scales,      // [N, K/groupsize]
    half* d_C,           // [M, N]
    int M, int N, int K,
    int groupsize,
    cudaStream_t stream,
    bool* used_gemv = nullptr)  // 输出: 是否使用了 GEMV
{
    const int GEMV_THRESHOLD = 64;  // M < 64 使用 GEMV

    if (M < GEMV_THRESHOLD) {
        // 使用 CUDA Core GEMV (小 batch)
        if (used_gemv) *used_gemv = true;

        wo::Params params(
            d_A,                              // act
            nullptr,                          // act_scale (W4A16 不需要)
            d_B,                              // weight
            d_scales,                         // scales
            nullptr,                          // zeros (SCALE_ONLY)
            nullptr,                          // bias
            d_C,                              // out
            1.0f,                             // alpha
            M, N, K,
            groupsize,                        // groupsize
            wo::KernelType::FP16Int4Groupwise,
            false                             // apply_alpha_in_advance
        );

        // 调用 GEMV kernel
        int arch = 90;  // SM90 = H100/H800
        wo::kernel_launcher(arch, params, stream);

    } else {
        // 使用 CUTLASS (大 batch)
        if (used_gemv) *used_gemv = false;

        static CutlassFpAIntBGemmRunner<half, cutlass::uint4b_t,
            cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> runner;

        // 选择最优配置
        static bool config_initialized = false;
        static CutlassGemmConfig best_config;

        if (!config_initialized) {
            auto configs = runner.getConfigs();
            best_config = selectBestConfig(configs, M, N, K);
            config_initialized = true;
        }

        // 分配 workspace
        size_t ws_bytes = runner.getWorkspaceSize(M, N, K);
        char* d_ws = nullptr;
        if (ws_bytes > 0) {
            CUDA_CHECK(cudaMalloc(&d_ws, ws_bytes));
        }

        // 运行 CUTLASS kernel
        runner.gemm(
            d_A,
            reinterpret_cast<cutlass::uint4b_t*>(d_B),
            d_scales,
            nullptr,  // zero points
            nullptr,  // biases
            1.0f,     // alpha
            d_C,
            M, N, K,
            groupsize,
            best_config,
            d_ws,
            ws_bytes,
            stream
        );

        if (d_ws) CUDA_CHECK(cudaFree(d_ws));
    }
}

// 性能测试
float benchmark_gemm(
    half* d_A, uint8_t* d_B, half* d_scales, half* d_C,
    int M, int N, int K, int groupsize,
    cudaStream_t stream,
    int num_iters = 10)
{
    // 预热
    for (int i = 0; i < 3; i++) {
        run_w4a16_gemm(d_A, d_B, d_scales, d_C, M, N, K, groupsize, stream, nullptr);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 测速
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);
    for (int i = 0; i < num_iters; i++) {
        run_w4a16_gemm(d_A, d_B, d_scales, d_C, M, N, K, groupsize, stream, nullptr);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return time_ms / num_iters;
}

int main(int argc, char** argv)
{
    printf("=== W4A16 完整测试 (GEMV + CUTLASS) ===\n\n");

    // 检查 GPU
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n\n", prop.major, prop.minor);

    if (prop.major != 9 || prop.minor != 0) {
        fprintf(stderr, "此程序需要 H100/H800 GPU (SM 9.0)\n");
        fprintf(stderr, "当前 GPU: SM %d.%d\n", prop.major, prop.minor);
        return 1;
    }

    // 矩阵维度
    int M = (argc > 1) ? atoi(argv[1]) : 32;
    int N = (argc > 2) ? atoi(argv[2]) : 1024;
    int K = (argc > 3) ? atoi(argv[3]) : 1024;
    int groupsize = 128;

    printf("Matrix: M=%d, N=%d, K=%d, groupsize=%d\n", M, N, K, groupsize);
    printf("Threshold: M < 64 使用 GEMV, M >= 64 使用 CUTLASS\n\n");

    // 检查 GEMV 支持
    if (!wo::is_supported(90, wo::KernelType::FP16Int4Groupwise)) {
        fprintf(stderr, "FP16Int4Groupwise not supported on SM90!\n");
        return 1;
    }
    printf("✅ GEMV kernel supported\n\n");

    // 分配内存
    size_t size_A = M * K;
    size_t size_B = N * K / 2;
    size_t size_C = M * N;
    size_t size_scales = N * (K / groupsize);

    half *d_A, *d_C, *d_scales;
    uint8_t *d_B;

    printf("Allocating memory...\n");
    CUDA_CHECK(cudaMalloc(&d_A, size_A * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_scales, size_scales * sizeof(half)));

    // 初始化数据
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

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    printf("\n=== 运行 GEMM ===\n");

    bool used_gemv = false;
    run_w4a16_gemm(d_A, d_B, d_scales, d_C, M, N, K, groupsize, stream, &used_gemv);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    printf("使用的 kernel: %s\n", used_gemv ? "CUDA Core GEMV" : "CUTLASS TMA");
    printf("✅ GEMM completed!\n\n");

    // 检查输出
    printf("=== 检查输出 ===\n");
    std::vector<half> h_C(size_C);
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size_C * sizeof(half), cudaMemcpyDeviceToHost));

    int non_zero = 0;
    float sum = 0.0f;
    for (size_t i = 0; i < size_C; i++) {
        float val = __half2float(h_C[i]);
        if (val != 0.0f) {
            non_zero++;
            sum += val;
        }
    }

    printf("Non-zero elements: %d / %zu (%.2f%%)\n",
           non_zero, size_C, (non_zero * 100.0f) / size_C);
    printf("Average value: %.6f\n", sum / size_C);

    printf("\nFirst 10 values:\n");
    for (int i = 0; i < std::min(10, (int)size_C); i++) {
        printf("  C[%d] = %.6f\n", i, __half2float(h_C[i]));
    }

    // 性能测试
    printf("\n=== 性能测试 ===\n");
    float time_ms = benchmark_gemm(d_A, d_B, d_scales, d_C, M, N, K, groupsize, stream, 10);

    // 计算 TFLOPS
    double flops = 2.0 * M * N * K;  // FMA = 2 ops
    double tflops = (flops / (time_ms * 1e-3)) / 1e12;

    printf("Time: %.3f ms\n", time_ms);
    printf("Performance: %.2f TFLOPS\n", tflops);

    // 测试不同 batch sizes
    printf("\n=== 测试不同 Batch Sizes ===\n");
    printf("%-8s  %-12s  %-10s  %-12s\n", "M", "Kernel", "Time(ms)", "TFLOPS");
    printf("-------------------------------------------------------\n");

    int test_Ms[] = {1, 4, 16, 32, 64, 128, 256, 512};
    for (int test_M : test_Ms) {
        if (test_M > M && argc > 1) continue;  // Skip if larger than requested

        size_t test_size_A = test_M * K;
        size_t test_size_C = test_M * N;

        half *test_d_A, *test_d_C;
        CUDA_CHECK(cudaMalloc(&test_d_A, test_size_A * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&test_d_C, test_size_C * sizeof(half)));
        CUDA_CHECK(cudaMemcpy(test_d_A, h_A.data(), test_size_A * sizeof(half), cudaMemcpyHostToDevice));

        bool test_used_gemv = false;
        float test_time = benchmark_gemm(test_d_A, d_B, d_scales, test_d_C,
                                        test_M, N, K, groupsize, stream, 10);

        run_w4a16_gemm(test_d_A, d_B, d_scales, test_d_C, test_M, N, K, groupsize, stream, &test_used_gemv);

        double test_flops = 2.0 * test_M * N * K;
        double test_tflops = (test_flops / (test_time * 1e-3)) / 1e12;

        printf("%-8d  %-12s  %-10.3f  %-12.2f\n",
               test_M,
               test_used_gemv ? "GEMV" : "CUTLASS",
               test_time,
               test_tflops);

        CUDA_CHECK(cudaFree(test_d_A));
        CUDA_CHECK(cudaFree(test_d_C));
    }

    // 清理
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_scales));
    CUDA_CHECK(cudaStreamDestroy(stream));

    printf("\n✅ 测试完成!\n");
    return 0;
}
