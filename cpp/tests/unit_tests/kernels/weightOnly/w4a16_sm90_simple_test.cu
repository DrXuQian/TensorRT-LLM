/*
 * Simple test for W4A16 SM90 kernels using TensorRT-LLM's CutlassFpAIntBGemmRunner API
 */

#include <cuda_runtime.h>
#include <gtest/gtest.h>

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

class W4A16SM90Test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Check if we have SM90 GPU
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        if (prop.major * 10 + prop.minor < 90)
        {
            GTEST_SKIP() << "Test requires SM90+ GPU (Hopper)";
        }
    }
};

TEST_F(W4A16SM90Test, BasicGemm)
{
    printf("=== W4A16 SM90 Basic GEMM Test ===\n\n");

    // Matrix dimensions
    int M = 1024;
    int N = 2048;
    int K = 2048;
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
        h_scales[i] = __float2half(0.1f);  // Small scale for testing
    }

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scales, h_scales.data(), size_scales * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, size_C * sizeof(half)));

    // Create GEMM runner using TensorRT-LLM's API
    printf("Creating CutlassFpAIntBGemmRunner...\n");
    CutlassFpAIntBGemmRunner<half, cutlass::uint4b_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> runner;

    // Get workspace size
    size_t workspace_bytes = runner.getWorkspaceSize(M, N, K);
    printf("Workspace size: %zu bytes\n", workspace_bytes);

    char* d_workspace = nullptr;
    if (workspace_bytes > 0) {
        CUDA_CHECK(cudaMalloc(&d_workspace, workspace_bytes));
    }

    // Get available configs
    auto configs = runner.getConfigs();
    printf("Available configs: %zu\n", configs.size());

    CutlassGemmConfig gemm_config;
    if (!configs.empty()) {
        gemm_config = configs[0];  // Use first available config
        printf("Using config: CTA=%dx%dx%d\n",
               gemm_config.tile_config_sm90.cta_m,
               gemm_config.tile_config_sm90.cta_n,
               gemm_config.tile_config_sm90.cta_k);
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    printf("\nLaunching GEMM kernel...\n");

    // Call the GEMM using TensorRT-LLM's API
    runner.gemm(
        d_A,                                          // A
        reinterpret_cast<cutlass::uint4b_t*>(d_B),   // B (weights)
        d_scales,                                     // weight_scales
        nullptr,                                      // weight_zero_points
        nullptr,                                      // biases
        1.0f,                                         // alpha
        d_C,                                          // C (output)
        M, N, K,                                      // dimensions
        group_size,                                   // group_size
        gemm_config,                                  // config
        d_workspace,                                  // workspace
        workspace_bytes,                              // workspace size
        stream                                        // stream
    );

    cudaError_t err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << "Kernel launch failed: " << cudaGetErrorString(err);

    printf("Synchronizing stream...\n");
    err = cudaStreamSynchronize(stream);
    ASSERT_EQ(err, cudaSuccess) << "Stream synchronization failed: " << cudaGetErrorString(err);

    printf("✅ Kernel completed successfully!\n\n");

    // Check output
    std::vector<half> h_C(std::min(size_t(100), size_C));
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(half), cudaMemcpyDeviceToHost));

    printf("First 10 outputs:\n");
    int zero_count = 0, nonzero_count = 0;
    for (int i = 0; i < std::min(10, (int)h_C.size()); i++) {
        float val = __half2float(h_C[i]);
        printf("  C[%d] = %.6f\n", i, val);
        if (val == 0.0f) zero_count++;
        else nonzero_count++;
    }

    for (size_t i = 10; i < h_C.size(); i++) {
        float val = __half2float(h_C[i]);
        if (val == 0.0f) zero_count++;
        else nonzero_count++;
    }

    printf("\nStatistics (first 100):\n");
    printf("  Zero: %d\n", zero_count);
    printf("  Non-zero: %d\n", nonzero_count);

    // Expect some non-zero outputs
    EXPECT_GT(nonzero_count, 0) << "All outputs are zero - kernel may not have executed properly";

    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_scales));
    if (d_workspace) {
        CUDA_CHECK(cudaFree(d_workspace));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
