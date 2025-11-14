/*
 * W4A16 Hopper (SM90) Kernel Extraction Test
 *
 * This file provides wrapper functions for the FP16-INT4 Weight-Only Quantized GEMM kernel
 * specifically for Hopper architecture (SM90).
 */

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/atom/mma_atom.hpp>
#include <stdio.h>

// Must include launcher implementation AFTER cutlass headers
#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/launchers/fpA_intB_launcher_sm90.inl"

using namespace tensorrt_llm::kernels::cutlass_kernels_oss;

// Type aliases for commonly used configurations
using CTAShape128 = cute::Shape<cute::Int<128>, cute::Int<128>, cute::Int<128>>;
using CTAShape64 = cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<128>>;
using ClusterShape = cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>;

// Wrapper function for 128x128x128 CTA with TMA Warp Specialized Cooperative
extern "C" void w4a16_sm90_gemm_128(
    half const* A,
    cutlass::uint4b_t const* B,
    half const* weight_scales,
    half const* weight_zero_points,
    half const* biases,
    float const alpha,
    half* C,
    int m,
    int n,
    int k,
    int const group_size,
    tensorrt_llm::cutlass_extensions::CutlassGemmConfig gemm_config,
    char* workspace,
    size_t workspace_bytes,
    cudaStream_t stream,
    int* occupancy = nullptr)
{
    sm90_generic_mixed_gemm_kernelLauncher<
        half,
        cutlass::uint4b_t,
        half,
        half,
        half,
        cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY,
        tensorrt_llm::cutlass_extensions::EpilogueOpBias,
        CTAShape128,
        ClusterShape,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative,
        cutlass::epilogue::TmaWarpSpecializedCooperative
    >(A, B, weight_scales, weight_zero_points, biases, alpha, C, m, n, k, group_size,
      gemm_config, workspace, workspace_bytes, stream, occupancy);
}

// Wrapper function for 64x128x128 CTA with TMA Warp Specialized Pingpong
extern "C" void w4a16_sm90_gemm_64(
    half const* A,
    cutlass::uint4b_t const* B,
    half const* weight_scales,
    half const* weight_zero_points,
    half const* biases,
    float const alpha,
    half* C,
    int m,
    int n,
    int k,
    int const group_size,
    tensorrt_llm::cutlass_extensions::CutlassGemmConfig gemm_config,
    char* workspace,
    size_t workspace_bytes,
    cudaStream_t stream,
    int* occupancy = nullptr)
{
    sm90_generic_mixed_gemm_kernelLauncher<
        half,
        cutlass::uint4b_t,
        half,
        half,
        half,
        cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY,
        tensorrt_llm::cutlass_extensions::EpilogueOpBias,
        CTAShape64,
        ClusterShape,
        cutlass::gemm::KernelTmaWarpSpecializedPingpong,
        cutlass::epilogue::TmaWarpSpecialized
    >(A, B, weight_scales, weight_zero_points, biases, alpha, C, m, n, k, group_size,
      gemm_config, workspace, workspace_bytes, stream, occupancy);
}

// Simple test function
extern "C" void test_w4a16_sm90_kernel()
{
    printf("W4A16 SM90 Hopper kernel compiled successfully!\n");
    printf("Available kernel variants:\n");
    printf("  - w4a16_sm90_gemm_128: 128x128x128 CTA with TMA Cooperative\n");
    printf("  - w4a16_sm90_gemm_64:  64x128x128 CTA with TMA Pingpong\n");
}
