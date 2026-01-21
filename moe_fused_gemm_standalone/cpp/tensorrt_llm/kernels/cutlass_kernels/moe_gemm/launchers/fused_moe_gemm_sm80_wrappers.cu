/*
 * Standalone wrapper implementations for SM80 fused MoE GEMM kernels (FP16).
 */

#include "tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/fused_moe_gemm_sm80_wrappers.h"
#include "tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/fused_moe_gemm_launcher_sm80.inl"
#include <limits>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::cutlass_kernels_oss
{
namespace
{
using LauncherFn = void (*)(cutlass::half_t const*, cutlass::half_t const*, cutlass::half_t const*, bool,
    cutlass::half_t*, int64_t const*, int64_t, int64_t, int64_t, int, int, cudaStream_t, int*);

template <int MaxTileM, int TileN, int TileK, int Stages, typename EpilogueTag>
void launch_config(cutlass::half_t const* A, cutlass::half_t const* B, cutlass::half_t const* biases,
    bool bias_is_broadcast, cutlass::half_t* C, int64_t const* total_tokens_including_expert, int64_t num_rows,
    int64_t gemm_n, int64_t gemm_k, int num_experts, int multi_processor_count, cudaStream_t stream,
    int* kernel_occupancy)
{
    sm80_generic_fused_moe_gemm_kernelLauncher<cutlass::half_t, cutlass::half_t, MaxTileM, TileN, TileK, Stages,
        EpilogueTag>(A, B, biases, bias_is_broadcast, C, total_tokens_including_expert, num_rows, gemm_n, gemm_k,
        num_experts, multi_processor_count, stream, kernel_occupancy);
}

struct FusedConfig
{
    int tile_m;
    int tile_n;
    int tile_k;
    int stages;
    LauncherFn launcher;
};

template <typename EpilogueTag>
const FusedConfig* select_config(int64_t num_rows, int64_t gemm_n, int64_t gemm_k)
{
    static const FusedConfig kConfigs[] = {
        {16, 128, 64, 2, &launch_config<16, 128, 64, 2, EpilogueTag>},
        {16, 128, 64, 3, &launch_config<16, 128, 64, 3, EpilogueTag>},
        {16, 256, 64, 2, &launch_config<16, 256, 64, 2, EpilogueTag>},
        {16, 256, 64, 3, &launch_config<16, 256, 64, 3, EpilogueTag>},
        {32, 128, 64, 2, &launch_config<32, 128, 64, 2, EpilogueTag>},
        {32, 128, 64, 3, &launch_config<32, 128, 64, 3, EpilogueTag>},
        {64, 128, 64, 2, &launch_config<64, 128, 64, 2, EpilogueTag>},
        {64, 128, 64, 3, &launch_config<64, 128, 64, 3, EpilogueTag>},
        {128, 128, 64, 2, &launch_config<128, 128, 64, 2, EpilogueTag>},
        {128, 128, 64, 3, &launch_config<128, 128, 64, 3, EpilogueTag>},
    };

    int64_t best_cost = std::numeric_limits<int64_t>::max();
    FusedConfig const* best = nullptr;
    for (auto const& cfg : kConfigs)
    {
        if ((gemm_n % cfg.tile_n) != 0 || (gemm_k % cfg.tile_k) != 0)
        {
            continue;
        }
        int64_t const tiles_m = (num_rows + cfg.tile_m - 1) / cfg.tile_m;
        int64_t const tiles_n = (gemm_n + cfg.tile_n - 1) / cfg.tile_n;
        int64_t cost = tiles_m * tiles_n;
        if (gemm_k >= 4096 && cfg.stages == 2)
        {
            cost += cost / 16;
        }
        if (gemm_k < 2048 && cfg.stages == 3)
        {
            cost += cost / 16;
        }
        if (cost < best_cost)
        {
            best_cost = cost;
            best = &cfg;
            continue;
        }
        if (cost == best_cost && best != nullptr)
        {
            if (cfg.tile_m > best->tile_m || (cfg.tile_m == best->tile_m && cfg.tile_n > best->tile_n))
            {
                best = &cfg;
            }
        }
    }
    return best;
}
} // namespace

void sm80_fused_moe_fc1_swiglu_fp16(cutlass::half_t const* A, cutlass::half_t const* B,
    cutlass::half_t const* biases, bool bias_is_broadcast, cutlass::half_t* C,
    int64_t const* total_tokens_including_expert, int64_t num_rows, int64_t gemm_n, int64_t gemm_k, int num_experts,
    int multi_processor_count, cudaStream_t stream, int* kernel_occupancy)
{
    using EpilogueTag = cutlass_extensions::EpilogueOpDefaultSilu;
    auto const* cfg = select_config<EpilogueTag>(num_rows, gemm_n, gemm_k);
    TLLM_CHECK_WITH_INFO(cfg != nullptr,
        "No valid fused MoE GEMM config for FC1 (gemm_n=%ld, gemm_k=%ld)", gemm_n, gemm_k);
    cfg->launcher(A, B, biases, bias_is_broadcast, C, total_tokens_including_expert, num_rows, gemm_n, gemm_k,
        num_experts, multi_processor_count, stream, kernel_occupancy);
}

void sm80_fused_moe_fc2_identity_fp16(cutlass::half_t const* A, cutlass::half_t const* B,
    cutlass::half_t const* biases, bool bias_is_broadcast, cutlass::half_t* C,
    int64_t const* total_tokens_including_expert, int64_t num_rows, int64_t gemm_n, int64_t gemm_k, int num_experts,
    int multi_processor_count, cudaStream_t stream, int* kernel_occupancy)
{
    using EpilogueTag = cutlass_extensions::EpilogueOpDefault;
    auto const* cfg = select_config<EpilogueTag>(num_rows, gemm_n, gemm_k);
    TLLM_CHECK_WITH_INFO(cfg != nullptr,
        "No valid fused MoE GEMM config for FC2 (gemm_n=%ld, gemm_k=%ld)", gemm_n, gemm_k);
    cfg->launcher(A, B, biases, bias_is_broadcast, C, total_tokens_including_expert, num_rows, gemm_n, gemm_k,
        num_experts, multi_processor_count, stream, kernel_occupancy);
}
} // namespace kernels::cutlass_kernels_oss

TRTLLM_NAMESPACE_END
