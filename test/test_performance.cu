// 性能测试 - 验证提取的 TensorRT-LLM kernel 性能
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <chrono>
#include <vector>
#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"

using namespace tensorrt_llm::kernels::cutlass_kernels;
using namespace std::chrono;

int main() {
    std::cout << "\n===== TensorRT-LLM FP16-INT4 Kernel 性能测试 =====\n" << std::endl;

    // 检查 GPU
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int sm = prop.major * 10 + prop.minor;
    std::cout << "GPU: " << prop.name << " (SM" << sm << ")" << std::endl;
    std::cout << "Peak Memory Bandwidth: " << prop.memoryBusWidth * prop.memoryClockRate * 2.0 / 8e6 << " GB/s" << std::endl;

    // 测试不同尺寸
    struct TestCase {
        int m, n, k;
        const char* desc;
    };

    std::vector<TestCase> test_cases = {
        {1, 4096, 4096, "Small (1x4096x4096)"},
        {1, 11008, 4096, "Medium (1x11008x4096)"},
        {32, 4096, 4096, "Batch 32 (32x4096x4096)"},
        {128, 4096, 4096, "Batch 128 (128x4096x4096)"},
    };

    const int group_size = 128;

    try {
        // 创建 kernel runner
        CutlassFpAIntBGemmRunner<half, cutlass::uint4b_t,
            cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> runner;
        std::cout << "✅ Kernel runner 创建成功\n" << std::endl;

        for (const auto& test : test_cases) {
            std::cout << "\n--- 测试 " << test.desc << " ---" << std::endl;
            std::cout << "M=" << test.m << ", N=" << test.n << ", K=" << test.k << std::endl;

            // 分配内存
            size_t weight_bytes = (test.k * test.n / 2);
            size_t scale_bytes = (test.k / group_size) * test.n * sizeof(half);

            half *d_input, *d_output, *d_scales;
            uint8_t *d_weights;

            cudaMalloc(&d_input, test.m * test.k * sizeof(half));
            cudaMalloc(&d_weights, weight_bytes);
            cudaMalloc(&d_scales, scale_bytes);
            cudaMalloc(&d_output, test.m * test.n * sizeof(half));

            // 初始化随机数据
            cudaMemset(d_weights, 0x88, weight_bytes);
            cudaMemset(d_scales, 0, scale_bytes);
            cudaMemset(d_input, 0, test.m * test.k * sizeof(half));
            cudaMemset(d_output, 0, test.m * test.n * sizeof(half));

            // 获取 workspace
            size_t workspace_size = runner.getWorkspaceSize(test.m, test.n, test.k);
            void* workspace = nullptr;
            if (workspace_size > 0) {
                cudaMalloc(&workspace, workspace_size);
            }

            // 配置
            tensorrt_llm::cutlass_extensions::CutlassGemmConfig config;
            config.tile_config_sm80 = tensorrt_llm::cutlass_extensions::CutlassTileConfig::CtaShape128x128x64_WarpShape128x32x64;
            config.split_k_style = tensorrt_llm::cutlass_extensions::SplitKStyle::NO_SPLIT_K;
            config.split_k_factor = 1;
            config.stages = 3;

            // Warmup
            for (int i = 0; i < 10; ++i) {
                runner.gemm(d_input, d_weights, d_scales, nullptr, nullptr,
                           1.0f, d_output, test.m, test.n, test.k, group_size, config,
                           (char*)workspace, workspace_size, 0);
            }
            cudaDeviceSynchronize();

            // 性能测试
            const int num_runs = 100;
            auto start = high_resolution_clock::now();

            for (int i = 0; i < num_runs; ++i) {
                runner.gemm(d_input, d_weights, d_scales, nullptr, nullptr,
                           1.0f, d_output, test.m, test.n, test.k, group_size, config,
                           (char*)workspace, workspace_size, 0);
            }
            cudaDeviceSynchronize();

            auto end = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(end - start).count();
            float avg_time_ms = duration / 1000.0f / num_runs;

            // 计算 GFLOPS (2 * M * N * K for GEMM)
            double flops = 2.0 * test.m * test.n * test.k;
            double gflops = (flops * num_runs) / (duration * 1e3);

            // 计算内存带宽使用
            size_t total_bytes = test.m * test.k * sizeof(half) +  // input
                                weight_bytes +                       // weights
                                scale_bytes +                        // scales
                                test.m * test.n * sizeof(half);     // output
            double bandwidth_gb = (total_bytes * num_runs) / (duration * 1e3);

            std::cout << "平均耗时: " << avg_time_ms << " ms" << std::endl;
            std::cout << "性能: " << gflops << " GFLOPS" << std::endl;
            std::cout << "有效带宽: " << bandwidth_gb << " GB/s" << std::endl;

            // 清理
            cudaFree(d_input);
            cudaFree(d_weights);
            cudaFree(d_scales);
            cudaFree(d_output);
            if (workspace) cudaFree(workspace);
        }

        std::cout << "\n✅✅✅ 性能测试完成！这是原始的 TensorRT-LLM kernel！" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "异常: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}