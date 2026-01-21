/*
 * Standalone wrapper implementations for SM80 fused MoE GEMM kernels (FP16).
 */

#include "tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/fused_moe_gemm_sm80_wrappers.h"
#include "tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/fused_moe_gemm_launcher_sm80.inl"

TRTLLM_NAMESPACE_BEGIN

namespace kernels::cutlass_kernels_oss
{
void sm80_fused_moe_fc1_swiglu_fp16(cutlass::half_t const* A, cutlass::half_t const* B,
    cutlass::half_t const* biases, bool bias_is_broadcast, cutlass::half_t* C,
    int64_t const* total_tokens_including_expert, int64_t num_rows, int64_t gemm_n, int64_t gemm_k, int num_experts,
    int multi_processor_count, cudaStream_t stream, int* kernel_occupancy)
{
    using EpilogueTag = cutlass_extensions::EpilogueOpDefaultSilu;
    sm80_generic_fused_moe_gemm_kernelLauncher<cutlass::half_t, cutlass::half_t, 128, 128, 64, 2, EpilogueTag>(
        A, B, biases, bias_is_broadcast, C, total_tokens_including_expert, num_rows, gemm_n, gemm_k, num_experts,
        multi_processor_count, stream, kernel_occupancy);
}

void sm80_fused_moe_fc2_identity_fp16(cutlass::half_t const* A, cutlass::half_t const* B,
    cutlass::half_t const* biases, bool bias_is_broadcast, cutlass::half_t* C,
    int64_t const* total_tokens_including_expert, int64_t num_rows, int64_t gemm_n, int64_t gemm_k, int num_experts,
    int multi_processor_count, cudaStream_t stream, int* kernel_occupancy)
{
    using EpilogueTag = cutlass_extensions::EpilogueOpDefault;
    sm80_generic_fused_moe_gemm_kernelLauncher<cutlass::half_t, cutlass::half_t, 128, 128, 64, 2, EpilogueTag>(
        A, B, biases, bias_is_broadcast, C, total_tokens_including_expert, num_rows, gemm_n, gemm_k, num_experts,
        multi_processor_count, stream, kernel_occupancy);
}
} // namespace kernels::cutlass_kernels_oss

TRTLLM_NAMESPACE_END
