/*
 * Standalone wrappers for SM80 fused MoE GEMM kernels (FP16).
 */

#pragma once

#include "tensorrt_llm/common/config.h"
#include <cuda_runtime_api.h>
#include <cutlass/half.h>
#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::cutlass_kernels_oss
{
struct Sm80FusedMoeGemmConfig
{
    int tile_m = 0;
    int tile_n = 0;
    int tile_k = 0;
    int stages = 0;
};

bool sm80_fused_moe_select_config_fc1(Sm80FusedMoeGemmConfig& out_config, int64_t const* total_tokens_including_expert,
    int num_experts, int64_t num_rows, int64_t gemm_n, int64_t gemm_k, int multi_processor_count);

bool sm80_fused_moe_select_config_fc2(Sm80FusedMoeGemmConfig& out_config, int64_t const* total_tokens_including_expert,
    int num_experts, int64_t num_rows, int64_t gemm_n, int64_t gemm_k, int multi_processor_count);

void sm80_fused_moe_fc1_swiglu_fp16_with_config(cutlass::half_t const* A, cutlass::half_t const* B,
    cutlass::half_t const* biases, bool bias_is_broadcast, cutlass::half_t* C,
    int64_t const* total_tokens_including_expert, int64_t num_rows, int64_t gemm_n, int64_t gemm_k, int num_experts,
    int multi_processor_count, cudaStream_t stream, int* kernel_occupancy, Sm80FusedMoeGemmConfig const& config);

void sm80_fused_moe_fc2_identity_fp16_with_config(cutlass::half_t const* A, cutlass::half_t const* B,
    cutlass::half_t const* biases, bool bias_is_broadcast, cutlass::half_t* C,
    int64_t const* total_tokens_including_expert, int64_t num_rows, int64_t gemm_n, int64_t gemm_k, int num_experts,
    int multi_processor_count, cudaStream_t stream, int* kernel_occupancy, Sm80FusedMoeGemmConfig const& config);

void sm80_fused_moe_fc1_swiglu_fp16(cutlass::half_t const* A, cutlass::half_t const* B,
    cutlass::half_t const* biases, bool bias_is_broadcast, cutlass::half_t* C,
    int64_t const* total_tokens_including_expert, int64_t num_rows, int64_t gemm_n, int64_t gemm_k, int num_experts,
    int multi_processor_count, cudaStream_t stream, int* kernel_occupancy);

void sm80_fused_moe_fc2_identity_fp16(cutlass::half_t const* A, cutlass::half_t const* B,
    cutlass::half_t const* biases, bool bias_is_broadcast, cutlass::half_t* C,
    int64_t const* total_tokens_including_expert, int64_t num_rows, int64_t gemm_n, int64_t gemm_k, int num_experts,
    int multi_processor_count, cudaStream_t stream, int* kernel_occupancy);
} // namespace kernels::cutlass_kernels_oss

TRTLLM_NAMESPACE_END
