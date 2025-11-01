// Simple test to verify kernel extraction works
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"

using namespace tensorrt_llm::kernels::cutlass_kernels;

int main() {
    std::cout << "Testing TensorRT-LLM FP16-INT4 kernel extraction" << std::endl;

    // Minimal test dimensions
    const int m = 1, n = 128, k = 128;
    const int group_size = 128;

    try {
        // Create kernel runner
        CutlassFpAIntBGemmRunner<half, cutlass::uint4b_t,
            cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> runner;

        std::cout << "✅ Kernel runner created successfully!" << std::endl;

        // Allocate GPU memory
        half *d_A, *d_C, *d_scales;
        uint8_t *d_B;

        cudaMalloc(&d_A, m * k * sizeof(half));
        cudaMalloc(&d_B, (k * n / 2));  // INT4 packed
        cudaMalloc(&d_scales, (k / group_size) * n * sizeof(half));
        cudaMalloc(&d_C, m * n * sizeof(half));

        // Get workspace size
        size_t ws_size = runner.getWorkspaceSize(m, n, k);
        std::cout << "Workspace size: " << ws_size << " bytes" << std::endl;

        // Cleanup
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_scales);
        cudaFree(d_C);

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "✅ Test passed!" << std::endl;
    return 0;
}