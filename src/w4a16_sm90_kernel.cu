/*
 * W4A16 Hopper (SM90) Kernel Extraction Test
 *
 * This file instantiates the FP16-INT4 Weight-Only Quantized GEMM kernel
 * specifically for Hopper architecture (SM90).
 */

#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm_template_sm90.h"
#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/launchers/fpA_intB_launcher_sm90.inl"

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/atom/mma_atom.hpp>

using namespace tensorrt_llm::kernels::cutlass_kernels_oss;

// Explicit template instantiation for FP16 activation, INT4 weight, FP16 output
// This is the W4A16 kernel for Hopper (SM90)
// Using fine-grained quantization with scale-only

// Test with 128x128x128 CTA shape
using CTAShape128 = cute::Shape<cute::Int<128>, cute::Int<128>, cute::Int<128>>;
using ClusterShape = cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>;

template void sm90_dispatch_mainloop_schedules<
    half,                                          // ActivationType
    cutlass::uint4b_t,                            // WeightType
    half,                                         // ScaleZeroType
    half,                                         // BiasType
    half,                                         // OutputType
    cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY,  // QuantOp
    tensorrt_llm::cutlass_extensions::EpilogueOpBias,    // EpilogueTag
    CTAShape128,                                  // CTAShape
    ClusterShape                                  // ClusterShape
>(
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
    int* occupancy
);

// Test with 64x128x128 CTA shape (smaller M dimension)
using CTAShape64 = cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<128>>;

template void sm90_dispatch_mainloop_schedules<
    half,
    cutlass::uint4b_t,
    half,
    half,
    half,
    cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY,
    tensorrt_llm::cutlass_extensions::EpilogueOpBias,
    CTAShape64,
    ClusterShape
>(
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
    int* occupancy
);

// Simple test function
extern "C" void test_w4a16_sm90_kernel()
{
    printf("W4A16 SM90 Hopper kernel instantiated successfully!\n");
}
