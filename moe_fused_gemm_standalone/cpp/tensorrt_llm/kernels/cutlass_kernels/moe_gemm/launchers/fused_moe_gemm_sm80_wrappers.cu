/*
 * Standalone wrapper implementations for SM80 fused MoE GEMM kernels (FP16).
 */

#include "tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/fused_moe_gemm_sm80_wrappers.h"
#include "tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/fused_moe_gemm_launcher_sm80.inl"
#include <limits>
#include <vector>

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

#define SM80_FUSED_CONFIG_LIST(X)                                                                                      \
    X(16, 128, 64, 2)                                                                                                  \
    X(16, 128, 64, 3)                                                                                                  \
    X(16, 128, 64, 4)                                                                                                  \
    X(16, 256, 64, 2)                                                                                                  \
    X(16, 256, 64, 3)                                                                                                  \
    X(16, 256, 64, 4)                                                                                                  \
    X(32, 128, 64, 2)                                                                                                  \
    X(32, 128, 64, 3)                                                                                                  \
    X(32, 128, 64, 4)                                                                                                  \
    X(64, 128, 64, 2)                                                                                                  \
    X(64, 128, 64, 3)                                                                                                  \
    X(64, 128, 64, 4)                                                                                                  \
    X(128, 128, 64, 2)                                                                                                 \
    X(128, 128, 64, 3)                                                                                                 \
    X(128, 128, 64, 4)

static const Sm80FusedMoeGemmConfig kAllConfigs[] = {
#define MAKE_CFG(m, n, k, s) {m, n, k, s},
    SM80_FUSED_CONFIG_LIST(MAKE_CFG)
#undef MAKE_CFG
};

template <typename EpilogueTag>
const FusedConfig* select_config(int64_t const* total_tokens_including_expert, int num_experts, int64_t num_rows,
    int64_t gemm_n, int64_t gemm_k, int multi_processor_count)
{
    static const FusedConfig kConfigs[] = {
#define MAKE_LAUNCH(m, n, k, s) {m, n, k, s, &launch_config<m, n, k, s, EpilogueTag>},
        SM80_FUSED_CONFIG_LIST(MAKE_LAUNCH)
#undef MAKE_LAUNCH
    };

    if (gemm_n <= 0 || gemm_k <= 0 || num_rows <= 0 || multi_processor_count <= 0)
    {
        return nullptr;
    }

    std::vector<int64_t> rows_per_expert;
    int64_t max_rows = 0;
    if (total_tokens_including_expert != nullptr && num_experts > 0)
    {
        rows_per_expert.resize(static_cast<size_t>(num_experts));
        int64_t prev = 0;
        for (int e = 0; e < num_experts; ++e)
        {
            int64_t const cur = total_tokens_including_expert[e];
            int64_t const rows = cur - prev;
            prev = cur;
            rows_per_expert[static_cast<size_t>(e)] = rows;
            if (rows > max_rows)
            {
                max_rows = rows;
            }
        }
    }
    else
    {
        max_rows = num_rows;
    }

    float best_score = 1.0f;
    int best_waves = std::numeric_limits<int>::max();
    int current_m_tile = 0;
    FusedConfig const* best = nullptr;
    for (auto const& cfg : kConfigs)
    {
        if ((gemm_n % cfg.tile_n) != 0 || (gemm_k % cfg.tile_k) != 0)
        {
            continue;
        }
        int occupancy = 0;
        cfg.launcher(nullptr, nullptr, nullptr, true, nullptr, nullptr, 0, 0, 0, 0, multi_processor_count, nullptr,
            &occupancy);
        if (occupancy == 0)
        {
            continue;
        }

        if (best != nullptr && max_rows < current_m_tile && current_m_tile < cfg.tile_m)
        {
            continue;
        }

        int64_t ctas_for_problem = 0;
        int64_t const tiles_n = (gemm_n + cfg.tile_n - 1) / cfg.tile_n;
        if (!rows_per_expert.empty())
        {
            for (int e = 0; e < num_experts; ++e)
            {
                int64_t const rows = rows_per_expert[static_cast<size_t>(e)];
                int64_t const tiles_m = (rows + cfg.tile_m - 1) / cfg.tile_m;
                ctas_for_problem += tiles_m * tiles_n;
            }
        }
        else
        {
            int64_t const tiles_m = (num_rows + cfg.tile_m - 1) / cfg.tile_m;
            ctas_for_problem = tiles_m * tiles_n;
        }

        int const ctas_per_wave = occupancy * multi_processor_count;
        if (ctas_per_wave <= 0)
        {
            continue;
        }
        int const num_waves_total = static_cast<int>((ctas_for_problem + ctas_per_wave - 1) / ctas_per_wave);
        float const num_waves_fractional = ctas_for_problem / static_cast<float>(ctas_per_wave);
        float const score = static_cast<float>(num_waves_total) - num_waves_fractional;

        float const score_slack = 0.1f;
        if (score < best_score || ((best_waves > num_waves_total) && (score < best_score + score_slack)))
        {
            best_score = score;
            best_waves = num_waves_total;
            best = &cfg;
            current_m_tile = cfg.tile_m;
        }
        else if (score == best_score && best != nullptr)
        {
            if (cfg.stages > best->stages || cfg.tile_m > best->tile_m)
            {
                best = &cfg;
                current_m_tile = cfg.tile_m;
                best_waves = num_waves_total;
            }
        }
    }
    return best;
}
} // namespace

size_t sm80_fused_moe_get_all_configs(Sm80FusedMoeGemmConfig const** configs)
{
    if (configs != nullptr)
    {
        *configs = kAllConfigs;
    }
    return sizeof(kAllConfigs) / sizeof(kAllConfigs[0]);
}

bool sm80_fused_moe_is_supported_config(Sm80FusedMoeGemmConfig const& config)
{
    for (size_t i = 0; i < sizeof(kAllConfigs) / sizeof(kAllConfigs[0]); ++i)
    {
        if (kAllConfigs[i].tile_m == config.tile_m && kAllConfigs[i].tile_n == config.tile_n
            && kAllConfigs[i].tile_k == config.tile_k && kAllConfigs[i].stages == config.stages)
        {
            return true;
        }
    }
    return false;
}

bool sm80_fused_moe_select_config_fc1(Sm80FusedMoeGemmConfig& out_config, int64_t const* total_tokens_including_expert,
    int num_experts, int64_t num_rows, int64_t gemm_n, int64_t gemm_k, int multi_processor_count)
{
    using EpilogueTag = cutlass_extensions::EpilogueOpDefaultSilu;
    if ((gemm_n % 64) != 0 || (gemm_k % 64) != 0)
    {
        return false;
    }
    auto const* cfg = select_config<EpilogueTag>(
        total_tokens_including_expert, num_experts, num_rows, gemm_n, gemm_k, multi_processor_count);
    if (cfg == nullptr)
    {
        return false;
    }
    out_config = {cfg->tile_m, cfg->tile_n, cfg->tile_k, cfg->stages};
    return true;
}

bool sm80_fused_moe_select_config_fc2(Sm80FusedMoeGemmConfig& out_config, int64_t const* total_tokens_including_expert,
    int num_experts, int64_t num_rows, int64_t gemm_n, int64_t gemm_k, int multi_processor_count)
{
    using EpilogueTag = cutlass_extensions::EpilogueOpDefault;
    if ((gemm_n % 64) != 0 || (gemm_k % 64) != 0)
    {
        return false;
    }
    auto const* cfg = select_config<EpilogueTag>(
        total_tokens_including_expert, num_experts, num_rows, gemm_n, gemm_k, multi_processor_count);
    if (cfg == nullptr)
    {
        return false;
    }
    out_config = {cfg->tile_m, cfg->tile_n, cfg->tile_k, cfg->stages};
    return true;
}

void sm80_fused_moe_fc1_swiglu_fp16_with_config(cutlass::half_t const* A, cutlass::half_t const* B,
    cutlass::half_t const* biases, bool bias_is_broadcast, cutlass::half_t* C,
    int64_t const* total_tokens_including_expert, int64_t num_rows, int64_t gemm_n, int64_t gemm_k, int num_experts,
    int multi_processor_count, cudaStream_t stream, int* kernel_occupancy, Sm80FusedMoeGemmConfig const& config)
{
    using EpilogueTag = cutlass_extensions::EpilogueOpDefaultSilu;
    TLLM_CHECK_WITH_INFO(config.tile_k == 64, "Unsupported tile_k=%d", config.tile_k);
    TLLM_CHECK_WITH_INFO(config.tile_n == 128 || config.tile_n == 256, "Unsupported tile_n=%d", config.tile_n);
    TLLM_CHECK_WITH_INFO(config.stages >= 2 && config.stages <= 4, "Unsupported stages=%d", config.stages);

    if (config.tile_m == 16 && config.tile_n == 128 && config.stages == 2)
        return launch_config<16, 128, 64, 2, EpilogueTag>(A, B, biases, bias_is_broadcast, C,
            total_tokens_including_expert, num_rows, gemm_n, gemm_k, num_experts, multi_processor_count, stream,
            kernel_occupancy);
    if (config.tile_m == 16 && config.tile_n == 128 && config.stages == 3)
        return launch_config<16, 128, 64, 3, EpilogueTag>(A, B, biases, bias_is_broadcast, C,
            total_tokens_including_expert, num_rows, gemm_n, gemm_k, num_experts, multi_processor_count, stream,
            kernel_occupancy);
    if (config.tile_m == 16 && config.tile_n == 128 && config.stages == 4)
        return launch_config<16, 128, 64, 4, EpilogueTag>(A, B, biases, bias_is_broadcast, C,
            total_tokens_including_expert, num_rows, gemm_n, gemm_k, num_experts, multi_processor_count, stream,
            kernel_occupancy);
    if (config.tile_m == 16 && config.tile_n == 256 && config.stages == 2)
        return launch_config<16, 256, 64, 2, EpilogueTag>(A, B, biases, bias_is_broadcast, C,
            total_tokens_including_expert, num_rows, gemm_n, gemm_k, num_experts, multi_processor_count, stream,
            kernel_occupancy);
    if (config.tile_m == 16 && config.tile_n == 256 && config.stages == 3)
        return launch_config<16, 256, 64, 3, EpilogueTag>(A, B, biases, bias_is_broadcast, C,
            total_tokens_including_expert, num_rows, gemm_n, gemm_k, num_experts, multi_processor_count, stream,
            kernel_occupancy);
    if (config.tile_m == 16 && config.tile_n == 256 && config.stages == 4)
        return launch_config<16, 256, 64, 4, EpilogueTag>(A, B, biases, bias_is_broadcast, C,
            total_tokens_including_expert, num_rows, gemm_n, gemm_k, num_experts, multi_processor_count, stream,
            kernel_occupancy);
    if (config.tile_m == 32 && config.tile_n == 128 && config.stages == 2)
        return launch_config<32, 128, 64, 2, EpilogueTag>(A, B, biases, bias_is_broadcast, C,
            total_tokens_including_expert, num_rows, gemm_n, gemm_k, num_experts, multi_processor_count, stream,
            kernel_occupancy);
    if (config.tile_m == 32 && config.tile_n == 128 && config.stages == 3)
        return launch_config<32, 128, 64, 3, EpilogueTag>(A, B, biases, bias_is_broadcast, C,
            total_tokens_including_expert, num_rows, gemm_n, gemm_k, num_experts, multi_processor_count, stream,
            kernel_occupancy);
    if (config.tile_m == 32 && config.tile_n == 128 && config.stages == 4)
        return launch_config<32, 128, 64, 4, EpilogueTag>(A, B, biases, bias_is_broadcast, C,
            total_tokens_including_expert, num_rows, gemm_n, gemm_k, num_experts, multi_processor_count, stream,
            kernel_occupancy);
    if (config.tile_m == 64 && config.tile_n == 128 && config.stages == 2)
        return launch_config<64, 128, 64, 2, EpilogueTag>(A, B, biases, bias_is_broadcast, C,
            total_tokens_including_expert, num_rows, gemm_n, gemm_k, num_experts, multi_processor_count, stream,
            kernel_occupancy);
    if (config.tile_m == 64 && config.tile_n == 128 && config.stages == 3)
        return launch_config<64, 128, 64, 3, EpilogueTag>(A, B, biases, bias_is_broadcast, C,
            total_tokens_including_expert, num_rows, gemm_n, gemm_k, num_experts, multi_processor_count, stream,
            kernel_occupancy);
    if (config.tile_m == 64 && config.tile_n == 128 && config.stages == 4)
        return launch_config<64, 128, 64, 4, EpilogueTag>(A, B, biases, bias_is_broadcast, C,
            total_tokens_including_expert, num_rows, gemm_n, gemm_k, num_experts, multi_processor_count, stream,
            kernel_occupancy);
    if (config.tile_m == 128 && config.tile_n == 128 && config.stages == 2)
        return launch_config<128, 128, 64, 2, EpilogueTag>(A, B, biases, bias_is_broadcast, C,
            total_tokens_including_expert, num_rows, gemm_n, gemm_k, num_experts, multi_processor_count, stream,
            kernel_occupancy);
    if (config.tile_m == 128 && config.tile_n == 128 && config.stages == 3)
        return launch_config<128, 128, 64, 3, EpilogueTag>(A, B, biases, bias_is_broadcast, C,
            total_tokens_including_expert, num_rows, gemm_n, gemm_k, num_experts, multi_processor_count, stream,
            kernel_occupancy);
    if (config.tile_m == 128 && config.tile_n == 128 && config.stages == 4)
        return launch_config<128, 128, 64, 4, EpilogueTag>(A, B, biases, bias_is_broadcast, C,
            total_tokens_including_expert, num_rows, gemm_n, gemm_k, num_experts, multi_processor_count, stream,
            kernel_occupancy);

    TLLM_CHECK_WITH_INFO(false, "Unsupported fused MoE config for FC1");
}

void sm80_fused_moe_fc2_identity_fp16_with_config(cutlass::half_t const* A, cutlass::half_t const* B,
    cutlass::half_t const* biases, bool bias_is_broadcast, cutlass::half_t* C,
    int64_t const* total_tokens_including_expert, int64_t num_rows, int64_t gemm_n, int64_t gemm_k, int num_experts,
    int multi_processor_count, cudaStream_t stream, int* kernel_occupancy, Sm80FusedMoeGemmConfig const& config)
{
    using EpilogueTag = cutlass_extensions::EpilogueOpDefault;
    TLLM_CHECK_WITH_INFO(config.tile_k == 64, "Unsupported tile_k=%d", config.tile_k);
    TLLM_CHECK_WITH_INFO(config.tile_n == 128 || config.tile_n == 256, "Unsupported tile_n=%d", config.tile_n);
    TLLM_CHECK_WITH_INFO(config.stages >= 2 && config.stages <= 4, "Unsupported stages=%d", config.stages);

    if (config.tile_m == 16 && config.tile_n == 128 && config.stages == 2)
        return launch_config<16, 128, 64, 2, EpilogueTag>(A, B, biases, bias_is_broadcast, C,
            total_tokens_including_expert, num_rows, gemm_n, gemm_k, num_experts, multi_processor_count, stream,
            kernel_occupancy);
    if (config.tile_m == 16 && config.tile_n == 128 && config.stages == 3)
        return launch_config<16, 128, 64, 3, EpilogueTag>(A, B, biases, bias_is_broadcast, C,
            total_tokens_including_expert, num_rows, gemm_n, gemm_k, num_experts, multi_processor_count, stream,
            kernel_occupancy);
    if (config.tile_m == 16 && config.tile_n == 128 && config.stages == 4)
        return launch_config<16, 128, 64, 4, EpilogueTag>(A, B, biases, bias_is_broadcast, C,
            total_tokens_including_expert, num_rows, gemm_n, gemm_k, num_experts, multi_processor_count, stream,
            kernel_occupancy);
    if (config.tile_m == 16 && config.tile_n == 256 && config.stages == 2)
        return launch_config<16, 256, 64, 2, EpilogueTag>(A, B, biases, bias_is_broadcast, C,
            total_tokens_including_expert, num_rows, gemm_n, gemm_k, num_experts, multi_processor_count, stream,
            kernel_occupancy);
    if (config.tile_m == 16 && config.tile_n == 256 && config.stages == 3)
        return launch_config<16, 256, 64, 3, EpilogueTag>(A, B, biases, bias_is_broadcast, C,
            total_tokens_including_expert, num_rows, gemm_n, gemm_k, num_experts, multi_processor_count, stream,
            kernel_occupancy);
    if (config.tile_m == 16 && config.tile_n == 256 && config.stages == 4)
        return launch_config<16, 256, 64, 4, EpilogueTag>(A, B, biases, bias_is_broadcast, C,
            total_tokens_including_expert, num_rows, gemm_n, gemm_k, num_experts, multi_processor_count, stream,
            kernel_occupancy);
    if (config.tile_m == 32 && config.tile_n == 128 && config.stages == 2)
        return launch_config<32, 128, 64, 2, EpilogueTag>(A, B, biases, bias_is_broadcast, C,
            total_tokens_including_expert, num_rows, gemm_n, gemm_k, num_experts, multi_processor_count, stream,
            kernel_occupancy);
    if (config.tile_m == 32 && config.tile_n == 128 && config.stages == 3)
        return launch_config<32, 128, 64, 3, EpilogueTag>(A, B, biases, bias_is_broadcast, C,
            total_tokens_including_expert, num_rows, gemm_n, gemm_k, num_experts, multi_processor_count, stream,
            kernel_occupancy);
    if (config.tile_m == 32 && config.tile_n == 128 && config.stages == 4)
        return launch_config<32, 128, 64, 4, EpilogueTag>(A, B, biases, bias_is_broadcast, C,
            total_tokens_including_expert, num_rows, gemm_n, gemm_k, num_experts, multi_processor_count, stream,
            kernel_occupancy);
    if (config.tile_m == 64 && config.tile_n == 128 && config.stages == 2)
        return launch_config<64, 128, 64, 2, EpilogueTag>(A, B, biases, bias_is_broadcast, C,
            total_tokens_including_expert, num_rows, gemm_n, gemm_k, num_experts, multi_processor_count, stream,
            kernel_occupancy);
    if (config.tile_m == 64 && config.tile_n == 128 && config.stages == 3)
        return launch_config<64, 128, 64, 3, EpilogueTag>(A, B, biases, bias_is_broadcast, C,
            total_tokens_including_expert, num_rows, gemm_n, gemm_k, num_experts, multi_processor_count, stream,
            kernel_occupancy);
    if (config.tile_m == 64 && config.tile_n == 128 && config.stages == 4)
        return launch_config<64, 128, 64, 4, EpilogueTag>(A, B, biases, bias_is_broadcast, C,
            total_tokens_including_expert, num_rows, gemm_n, gemm_k, num_experts, multi_processor_count, stream,
            kernel_occupancy);
    if (config.tile_m == 128 && config.tile_n == 128 && config.stages == 2)
        return launch_config<128, 128, 64, 2, EpilogueTag>(A, B, biases, bias_is_broadcast, C,
            total_tokens_including_expert, num_rows, gemm_n, gemm_k, num_experts, multi_processor_count, stream,
            kernel_occupancy);
    if (config.tile_m == 128 && config.tile_n == 128 && config.stages == 3)
        return launch_config<128, 128, 64, 3, EpilogueTag>(A, B, biases, bias_is_broadcast, C,
            total_tokens_including_expert, num_rows, gemm_n, gemm_k, num_experts, multi_processor_count, stream,
            kernel_occupancy);
    if (config.tile_m == 128 && config.tile_n == 128 && config.stages == 4)
        return launch_config<128, 128, 64, 4, EpilogueTag>(A, B, biases, bias_is_broadcast, C,
            total_tokens_including_expert, num_rows, gemm_n, gemm_k, num_experts, multi_processor_count, stream,
            kernel_occupancy);

    TLLM_CHECK_WITH_INFO(false, "Unsupported fused MoE config for FC2");
}

void sm80_fused_moe_fc1_swiglu_fp16(cutlass::half_t const* A, cutlass::half_t const* B,
    cutlass::half_t const* biases, bool bias_is_broadcast, cutlass::half_t* C,
    int64_t const* total_tokens_including_expert, int64_t num_rows, int64_t gemm_n, int64_t gemm_k, int num_experts,
    int multi_processor_count, cudaStream_t stream, int* kernel_occupancy)
{
    using EpilogueTag = cutlass_extensions::EpilogueOpDefaultSilu;
    Sm80FusedMoeGemmConfig cfg{};
    TLLM_CHECK_WITH_INFO(sm80_fused_moe_select_config_fc1(
                             cfg, total_tokens_including_expert, num_experts, num_rows, gemm_n, gemm_k,
                             multi_processor_count),
        "No valid fused MoE GEMM config for FC1 (gemm_n=%ld, gemm_k=%ld)", gemm_n, gemm_k);
    sm80_fused_moe_fc1_swiglu_fp16_with_config(A, B, biases, bias_is_broadcast, C, total_tokens_including_expert,
        num_rows, gemm_n, gemm_k, num_experts, multi_processor_count, stream, kernel_occupancy, cfg);
}

void sm80_fused_moe_fc2_identity_fp16(cutlass::half_t const* A, cutlass::half_t const* B,
    cutlass::half_t const* biases, bool bias_is_broadcast, cutlass::half_t* C,
    int64_t const* total_tokens_including_expert, int64_t num_rows, int64_t gemm_n, int64_t gemm_k, int num_experts,
    int multi_processor_count, cudaStream_t stream, int* kernel_occupancy)
{
    using EpilogueTag = cutlass_extensions::EpilogueOpDefault;
    Sm80FusedMoeGemmConfig cfg{};
    TLLM_CHECK_WITH_INFO(sm80_fused_moe_select_config_fc2(
                             cfg, total_tokens_including_expert, num_experts, num_rows, gemm_n, gemm_k,
                             multi_processor_count),
        "No valid fused MoE GEMM config for FC2 (gemm_n=%ld, gemm_k=%ld)", gemm_n, gemm_k);
    sm80_fused_moe_fc2_identity_fp16_with_config(A, B, biases, bias_is_broadcast, C, total_tokens_including_expert,
        num_rows, gemm_n, gemm_k, num_experts, multi_processor_count, stream, kernel_occupancy, cfg);
}
} // namespace kernels::cutlass_kernels_oss

TRTLLM_NAMESPACE_END
