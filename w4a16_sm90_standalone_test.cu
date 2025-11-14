/*
 * Standalone test for W4A16 SM90 kernels using TensorRT-LLM's CutlassFpAIntBGemmRunner API
 *
 * This is the CORRECT way to use the W4A16 kernels - through the TensorRT-LLM API,
 * not by calling the low-level launcher functions directly.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "cutlass/numeric_types.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"

#include <cstdio>
#include <iostream>
#include <random>
#include <vector>

using namespace tensorrt_llm::kernels::cutlass_kernels;
using namespace tensorrt_llm::cutlass_extensions;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

int main(int argc, char** argv)
{
    printf("=== W4A16 SM90 Standalone Test using TensorRT-LLM API ===\n\n");

    // Check if we have SM90+ GPU
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n\n", prop.major, prop.minor);

    if (prop.major * 10 + prop.minor < 90)
    {
        fprintf(stderr, "This test requires SM90+ GPU (Hopper)\n");
        fprintf(stderr, "Current GPU is SM%d%d\n", prop.major, prop.minor);
        return 1;
    }

    // Matrix dimensions
    int M = (argc > 1) ? atoi(argv[1]) : 1024;
    int N = (argc > 2) ? atoi(argv[2]) : 2048;
    int K = (argc > 3) ? atoi(argv[3]) : 2048;
    int group_size = 128;

    printf("Config: M=%d, N=%d, K=%d, group_size=%d\n\n", M, N, K, group_size);

    // Calculate sizes
    size_t size_A = M * K;
    size_t size_B = N * K / 2;  // INT4: 2 values per byte
    size_t size_C = M * N;
    size_t size_scales = N * (K / group_size);

    // Allocate device memory
    half *d_A, *d_C, *d_scales;
    uint8_t *d_B;

    printf("Allocating device memory...\n");
    CUDA_CHECK(cudaMalloc(&d_A, size_A * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_scales, size_scales * sizeof(half)));

    // Initialize data
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

    printf("\n=== Creating CutlassFpAIntBGemmRunner (TensorRT-LLM API) ===\n");

    // This is the KEY DIFFERENCE:
    // We use TensorRT-LLM's CutlassFpAIntBGemmRunner class,
    // which handles all the initialization and configuration properly
    CutlassFpAIntBGemmRunner<half, cutlass::uint4b_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> runner;

    printf("Runner created successfully!\n");

    // Get workspace size
    size_t workspace_bytes = runner.getWorkspaceSize(M, N, K);
    printf("Workspace size: %zu bytes (%.2f MB)\n", workspace_bytes, workspace_bytes / (1024.0 * 1024.0));

    char* d_workspace = nullptr;
    if (workspace_bytes > 0) {
        CUDA_CHECK(cudaMalloc(&d_workspace, workspace_bytes));
        printf("Workspace allocated\n");
    }

    // Get available configs
    auto configs = runner.getConfigs();
    printf("Available configs: %zu\n", configs.size());

    CutlassGemmConfig gemm_config;
    if (!configs.empty()) {
        gemm_config = configs[0];  // Use first available config
        printf("Using config #0 (enum value: %d)\n", static_cast<int>(gemm_config.tile_config_sm90));
    } else {
        printf("No configs available - using default\n");
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    printf("\n=== Launching GEMM using TensorRT-LLM API ===\n");

    try {
        // This is the CORRECT API call - let TensorRT-LLM handle everything
        runner.gemm(
            d_A,                                          // A (activations)
            reinterpret_cast<cutlass::uint4b_t*>(d_B),   // B (INT4 weights)
            d_scales,                                     // weight_scales
            nullptr,                                      // weight_zero_points (not used)
            nullptr,                                      // biases (not used)
            1.0f,                                         // alpha
            d_C,                                          // C (output)
            M, N, K,                                      // dimensions
            group_size,                                   // group_size for fine-grained quantization
            gemm_config,                                  // GEMM config
            d_workspace,                                  // workspace pointer
            workspace_bytes,                              // workspace size
            stream                                        // CUDA stream
        );

        printf("GEMM call returned successfully!\n");

    } catch (const std::exception& e) {
        fprintf(stderr, "\n❌ Exception during GEMM: %s\n", e.what());
        return 1;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "❌ Kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("No kernel launch errors\n");

    printf("Synchronizing stream...\n");
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "❌ Stream synchronization failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("✅ Stream synchronized successfully!\n\n");

    // Check output
    printf("=== Checking Results ===\n");
    std::vector<half> h_C(std::min(size_t(100), size_C));
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(half), cudaMemcpyDeviceToHost));

    printf("First 10 outputs:\n");
    int zero_count = 0, nonzero_count = 0;
    for (int i = 0; i < std::min(10, (int)h_C.size()); i++) {
        float val = __half2float(h_C[i]);
        printf("  C[%d] = %.6f\n", i, val);
        if (std::abs(val) < 1e-7f) zero_count++;
        else nonzero_count++;
    }

    for (size_t i = 10; i < h_C.size(); i++) {
        float val = __half2float(h_C[i]);
        if (std::abs(val) < 1e-7f) zero_count++;
        else nonzero_count++;
    }

    printf("\nStatistics (first 100 elements):\n");
    printf("  Zero: %d\n", zero_count);
    printf("  Non-zero: %d\n", nonzero_count);

    if (nonzero_count > 0) {
        printf("\n✅ SUCCESS: Kernel executed and produced non-zero outputs!\n");
    } else {
        printf("\n⚠️  WARNING: All outputs are zero - this may indicate an issue\n");
    }

    // Cleanup
    printf("\nCleaning up...\n");
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_scales));
    if (d_workspace) {
        CUDA_CHECK(cudaFree(d_workspace));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));

    printf("Done!\n");
    return 0;
}
