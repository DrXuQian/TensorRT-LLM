/*
 * W4A16 SM90 Performance Test with Config Selection
 * 测试不同配置的性能，并选择最优配置
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <random>
#include <vector>
#include <algorithm>
#include <cfloat>

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

// 配置信息结构
struct ConfigInfo {
    CutlassGemmConfig config;
    int index;
    float time_ms;
    bool success;
};

// 打印配置信息
void printConfig(const CutlassGemmConfig& cfg, int index) {
    // 提取 SM90 tile shape
    auto [tile_m, tile_n, tile_k] = enum_to_shape_tuple(cfg.tile_config_sm90);
    auto [cluster_m, cluster_n, cluster_k] = enum_to_shape_tuple(cfg.cluster_shape);

    printf("  [%d] Tile: M=%d N=%d K=%d",
           index, tile_m, tile_n, tile_k);

    printf(", Cluster: %dx%dx%d",
           cluster_m, cluster_n, cluster_k);

    if (cfg.mainloop_schedule == MainloopScheduleType::AUTO) {
        printf(", Sched: AUTO");
    } else if (cfg.mainloop_schedule == MainloopScheduleType::COOPERATIVE) {
        printf(", Sched: COOP");
    } else if (cfg.mainloop_schedule == MainloopScheduleType::PINGPONG) {
        printf(", Sched: PINGPONG");
    }

    printf("\n");
}

// 选择最优配置
CutlassGemmConfig selectBestConfig(
    const std::vector<CutlassGemmConfig>& configs,
    int M, int N, int K)
{
    // 启发式规则选择配置

    if (configs.empty()) {
        fprintf(stderr, "No configs available!\n");
        exit(1);
    }

    printf("\n=== Selecting Best Config ===\n");
    printf("Matrix size: M=%d, N=%d, K=%d\n\n", M, N, K);

    // 规则 1: 根据 M 大小选择 tile
    int preferred_m_tile = 64;
    if (M >= 256) {
        preferred_m_tile = 128;
    } else if (M >= 128) {
        preferred_m_tile = 128;
    } else {
        preferred_m_tile = 64;
    }

    // 规则 2: 优先选择非 CUDA kernel
    // 规则 3: tile M 尽量接近但不超过实际 M

    int best_idx = -1;
    int best_score = -1;

    for (size_t i = 0; i < configs.size(); i++) {
        const auto& cfg = configs[i];

        // 提取 tile shape
        auto [tile_m, tile_n, tile_k] = enum_to_shape_tuple(cfg.tile_config_sm90);

        // 跳过无效配置
        if (tile_m == 0) {
            continue;
        }

        int score = 0;

        // M tile 匹配度
        if (tile_m == preferred_m_tile) {
            score += 100;
        } else if (tile_m < preferred_m_tile) {
            score += 50;
        }

        // N tile 越大越好 (通常)
        score += tile_n / 32;

        // K tile 匹配 (SM90 uses K in bytes, typically 128B)
        if (tile_k >= 64) {
            score += 10;
        }

        // Cluster shape: 对于大矩阵，prefer larger clusters
        if (M >= 128 && N >= 128) {
            auto [cluster_m, cluster_n, cluster_k] = enum_to_shape_tuple(cfg.cluster_shape);
            score += (cluster_m + cluster_n) * 5;
        }

        // Schedule: COOPERATIVE 通常更好
        if (cfg.mainloop_schedule == MainloopScheduleType::COOPERATIVE) {
            score += 20;
        }

        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }

    if (best_idx < 0) {
        printf("Warning: No suitable config found, using first config\n");
        best_idx = 0;
    }

    printf("Selected config [%d] (score=%d):\n", best_idx, best_score);
    printConfig(configs[best_idx], best_idx);

    return configs[best_idx];
}

// Profile 所有配置 (可选)
ConfigInfo profileConfig(
    CutlassFpAIntBGemmRunner<half, cutlass::uint4b_t,
        cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY>& runner,
    const CutlassGemmConfig& config,
    int index,
    half* d_A, uint8_t* d_B, half* d_scales, half* d_C,
    int M, int N, int K, int group_size,
    char* d_workspace, size_t workspace_bytes,
    cudaStream_t stream)
{
    ConfigInfo info;
    info.config = config;
    info.index = index;
    info.success = false;
    info.time_ms = FLT_MAX;

    // 跳过无效配置
    auto [tile_m, tile_n, tile_k] = enum_to_shape_tuple(config.tile_config_sm90);
    if (tile_m == 0) {
        return info;
    }

    // 预热
    try {
        runner.gemm(
            d_A,
            reinterpret_cast<cutlass::uint4b_t*>(d_B),
            d_scales,
            nullptr, nullptr, 1.0f,
            d_C, M, N, K, group_size,
            config, d_workspace, workspace_bytes, stream
        );
        cudaStreamSynchronize(stream);
    } catch (...) {
        return info;  // 配置失败
    }

    // 测速
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int num_iters = 10;

    cudaEventRecord(start, stream);
    for (int i = 0; i < num_iters; i++) {
        runner.gemm(
            d_A,
            reinterpret_cast<cutlass::uint4b_t*>(d_B),
            d_scales,
            nullptr, nullptr, 1.0f,
            d_C, M, N, K, group_size,
            config, d_workspace, workspace_bytes, stream
        );
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float total_ms;
    cudaEventElapsedTime(&total_ms, start, stop);

    info.time_ms = total_ms / num_iters;
    info.success = true;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return info;
}

int main(int argc, char** argv)
{
    printf("=== W4A16 SM90 Performance Test ===\n\n");

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
    bool do_profiling = (argc > 4 && strcmp(argv[4], "--profile") == 0);

    printf("Matrix: M=%d, N=%d, K=%d, group_size=%d\n", M, N, K, group_size);
    printf("Profiling: %s\n\n", do_profiling ? "enabled" : "disabled");

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

    printf("\n=== Creating CutlassFpAIntBGemmRunner ===\n");

    CutlassFpAIntBGemmRunner<half, cutlass::uint4b_t,
        cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> runner;

    // 获取 workspace
    size_t workspace_bytes = runner.getWorkspaceSize(M, N, K);
    printf("Workspace: %.2f MB\n", workspace_bytes / (1024.0 * 1024.0));

    char* d_workspace = nullptr;
    if (workspace_bytes > 0) {
        CUDA_CHECK(cudaMalloc(&d_workspace, workspace_bytes));
    }

    // 获取所有配置
    auto configs = runner.getConfigs();
    printf("Available configs: %zu\n\n", configs.size());

    // 打印所有配置
    printf("=== All Configs ===\n");
    for (size_t i = 0; i < std::min(configs.size(), size_t(20)); i++) {
        printConfig(configs[i], i);
    }
    if (configs.size() > 20) {
        printf("  ... and %zu more configs\n", configs.size() - 20);
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    CutlassGemmConfig best_config;

    if (do_profiling && configs.size() > 1) {
        // Profile 模式: 测试所有配置
        printf("\n=== Profiling All Configs ===\n");
        printf("This may take a while...\n\n");

        std::vector<ConfigInfo> results;
        for (size_t i = 0; i < configs.size(); i++) {
            printf("\rProfiling config %zu/%zu...", i+1, configs.size());
            fflush(stdout);

            auto info = profileConfig(runner, configs[i], i,
                                     d_A, d_B, d_scales, d_C,
                                     M, N, K, group_size,
                                     d_workspace, workspace_bytes, stream);
            if (info.success) {
                results.push_back(info);
            }
        }
        printf("\n\n");

        // 排序并显示结果
        std::sort(results.begin(), results.end(),
                 [](const ConfigInfo& a, const ConfigInfo& b) {
                     return a.time_ms < b.time_ms;
                 });

        printf("=== Profiling Results (Top 10) ===\n");
        for (size_t i = 0; i < std::min(results.size(), size_t(10)); i++) {
            printf("[%d] %.3f ms - ", results[i].index, results[i].time_ms);
            printConfig(results[i].config, results[i].index);
        }

        if (!results.empty()) {
            best_config = results[0].config;
            printf("\nBest config: [%d] %.3f ms\n", results[0].index, results[0].time_ms);
        } else {
            printf("\nWarning: All configs failed, using heuristic selection\n");
            best_config = selectBestConfig(configs, M, N, K);
        }

    } else {
        // 启发式选择
        best_config = selectBestConfig(configs, M, N, K);
    }

    printf("\n=== Running GEMM with Selected Config ===\n");

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
            best_config,
            d_workspace,
            workspace_bytes,
            stream
        );

        CUDA_CHECK(cudaStreamSynchronize(stream));
        printf("✅ GEMM completed!\n\n");

    } catch (const std::exception& e) {
        fprintf(stderr, "Exception: %s\n", e.what());
        return 1;
    }

    // 检查输出
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
        printf("✅ PASSED\n");
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
