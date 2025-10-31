// 简单测试 - 验证提取的 TensorRT-LLM kernel
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"

using namespace tensorrt_llm::kernels::cutlass_kernels;

int main() {
    std::cout << "测试 TensorRT-LLM FP16-INT4 kernel 提取" << std::endl;

    // 最小维度测试
    const int m = 1, n = 128, k = 128;
    const int group_size = 128;

    try {
        // 创建 TensorRT-LLM kernel runner
        CutlassFpAIntBGemmRunner<half, cutlass::uint4b_t,
            cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> runner;

        std::cout << "✅ Kernel runner 创建成功!" << std::endl;

        // 分配 GPU 内存
        half *d_A, *d_C, *d_scales;
        uint8_t *d_B;

        cudaMalloc(&d_A, m * k * sizeof(half));
        cudaMalloc(&d_B, (k * n / 2));  // INT4 packed
        cudaMalloc(&d_scales, (k / group_size) * n * sizeof(half));
        cudaMalloc(&d_C, m * n * sizeof(half));

        // 获取 workspace
        size_t ws_size = runner.getWorkspaceSize(m, n, k);
        std::cout << "Workspace 大小: " << ws_size << " bytes" << std::endl;

        // 清理
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_scales);
        cudaFree(d_C);

    } catch (const std::exception& e) {
        std::cerr << "异常: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "✅ 测试通过!" << std::endl;
    return 0;
}