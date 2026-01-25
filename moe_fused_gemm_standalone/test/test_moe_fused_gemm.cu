/*
 * Standalone test for SM80 fused MoE GEMM kernels (FP16).
 */

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/fused_moe_gemm_sm80_wrappers.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace
{
struct Args
{
    int64_t num_tokens = 3823;
    int64_t hidden_size = 2048;
    int64_t inter_size = 768;
    int num_experts = 128;
    int experts_per_token = 8;
    bool run_fc1 = true;
    bool run_fc2 = true;
    bool verify = false;
    bool list_configs = false;
    bool force_config = false;
    tensorrt_llm::kernels::cutlass_kernels_oss::Sm80FusedMoeGemmConfig forced_config{};
};

void print_usage(char const* name)
{
    std::printf(
        "Usage: %s [--num_tokens=N] [--hidden_size=N] [--inter_size=N] [--num_experts=N] "
        "[--experts_per_token=N] [--op=fc1|fc2|both] [--verify] [--list_configs] "
        "[--config=tile_m,tile_n,tile_k,stages]\n",
        name);
}

bool parse_int64(char const* arg, char const* key, int64_t& out)
{
    size_t const len = std::strlen(key);
    if (std::strncmp(arg, key, len) != 0)
    {
        return false;
    }
    out = std::strtoll(arg + len, nullptr, 10);
    return true;
}

bool parse_int(char const* arg, char const* key, int& out)
{
    size_t const len = std::strlen(key);
    if (std::strncmp(arg, key, len) != 0)
    {
        return false;
    }
    out = std::strtol(arg + len, nullptr, 10);
    return true;
}

bool parse_config_spec(char const* spec,
    tensorrt_llm::kernels::cutlass_kernels_oss::Sm80FusedMoeGemmConfig& out)
{
    if (spec == nullptr || *spec == '\0')
    {
        return false;
    }
    std::string normalized(spec);
    for (char& ch : normalized)
    {
        if (ch == 'x' || ch == 'X')
        {
            ch = ',';
        }
    }
    int tile_m = 0;
    int tile_n = 0;
    int tile_k = 0;
    int stages = 0;
    if (std::sscanf(normalized.c_str(), "%d,%d,%d,%d", &tile_m, &tile_n, &tile_k, &stages) != 4)
    {
        return false;
    }
    if (tile_m <= 0 || tile_n <= 0 || tile_k <= 0 || stages <= 0)
    {
        return false;
    }
    out.tile_m = tile_m;
    out.tile_n = tile_n;
    out.tile_k = tile_k;
    out.stages = stages;
    return true;
}

void print_all_configs()
{
    using Config = tensorrt_llm::kernels::cutlass_kernels_oss::Sm80FusedMoeGemmConfig;
    Config const* configs = nullptr;
    size_t const count = tensorrt_llm::kernels::cutlass_kernels_oss::sm80_fused_moe_get_all_configs(&configs);
    std::printf("Supported SM80 fused configs (%zu):\n", count);
    for (size_t i = 0; i < count; ++i)
    {
        Config const& cfg = configs[i];
        std::printf("  %zu: tile_m=%d tile_n=%d tile_k=%d stages=%d\n", i, cfg.tile_m, cfg.tile_n, cfg.tile_k,
            cfg.stages);
    }
}

void parse_args(int argc, char** argv, Args& args)
{
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--help") == 0)
        {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (parse_int64(argv[i], "--num_tokens=", args.num_tokens))
        {
            continue;
        }
        if (parse_int64(argv[i], "--hidden_size=", args.hidden_size))
        {
            continue;
        }
        if (parse_int64(argv[i], "--inter_size=", args.inter_size))
        {
            continue;
        }
        if (parse_int(argv[i], "--num_experts=", args.num_experts))
        {
            continue;
        }
        if (parse_int(argv[i], "--experts_per_token=", args.experts_per_token))
        {
            continue;
        }
        if (std::strncmp(argv[i], "--op=", 5) == 0)
        {
            std::string op = argv[i] + 5;
            if (op == "fc1")
            {
                args.run_fc1 = true;
                args.run_fc2 = false;
            }
            else if (op == "fc2")
            {
                args.run_fc1 = false;
                args.run_fc2 = true;
            }
            else if (op == "both")
            {
                args.run_fc1 = true;
                args.run_fc2 = true;
            }
            else
            {
                std::fprintf(stderr, "Unknown op: %s\n", op.c_str());
                print_usage(argv[0]);
                std::exit(1);
            }
            continue;
        }
        if (std::strcmp(argv[i], "--list_configs") == 0)
        {
            args.list_configs = true;
            continue;
        }
        if (std::strncmp(argv[i], "--config=", 9) == 0)
        {
            if (!parse_config_spec(argv[i] + 9, args.forced_config))
            {
                std::fprintf(stderr,
                    "Invalid --config format. Expected --config=tile_m,tile_n,tile_k,stages (or 16x128x64x2).\n");
                print_usage(argv[0]);
                std::exit(1);
            }
            args.force_config = true;
            continue;
        }
        if (std::strcmp(argv[i], "--verify") == 0)
        {
            args.verify = true;
            continue;
        }
        std::fprintf(stderr, "Unknown argument: %s\n", argv[i]);
        print_usage(argv[0]);
        std::exit(1);
    }
}

__global__ void fill_half_kernel(cutlass::half_t* data, size_t n, float scale)
{
    size_t const idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float const val = static_cast<float>(idx % 97) * scale;
        data[idx] = cutlass::half_t(val);
    }
}

void fill_device_half(cutlass::half_t* data, size_t n, float scale, cudaStream_t stream)
{
    int const block = 256;
    int const grid = static_cast<int>((n + block - 1) / block);
    fill_half_kernel<<<grid, block, 0, stream>>>(data, n, scale);
}

void fill_host_half(cutlass::half_t* data, size_t n, float scale)
{
    for (size_t idx = 0; idx < n; ++idx)
    {
        float const val = static_cast<float>(idx % 97) * scale;
        data[idx] = cutlass::half_t(val);
    }
}

size_t checked_mul_size(size_t a, size_t b, char const* name)
{
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a)
    {
        std::fprintf(stderr, "Size overflow for %s\n", name);
        std::exit(1);
    }
    return a * b;
}

float checksum_first(cutlass::half_t const* data, size_t n)
{
    size_t const sample = std::min<size_t>(n, 1024);
    float sum = 0.0f;
    for (size_t i = 0; i < sample; ++i)
    {
        sum += static_cast<float>(data[i]);
    }
    return sum;
}

float silu(float x)
{
    return x / (1.0f + std::exp(-x));
}

struct DiffStats
{
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    size_t max_idx = 0;
};

bool find_nonfinite_half(cutlass::half_t const* data, size_t n, char const* name)
{
    for (size_t i = 0; i < n; ++i)
    {
        float const val = static_cast<float>(data[i]);
        if (!std::isfinite(val))
        {
            std::printf("%s has non-finite value at %zu: %f\n", name, i, val);
            return true;
        }
    }
    return false;
}

bool find_nonfinite_float(float const* data, size_t n, char const* name)
{
    for (size_t i = 0; i < n; ++i)
    {
        float const val = data[i];
        if (!std::isfinite(val))
        {
            std::printf("%s has non-finite value at %zu: %f\n", name, i, val);
            return true;
        }
    }
    return false;
}

DiffStats compare_half_to_float(cutlass::half_t const* gpu, float const* ref, size_t n)
{
    DiffStats stats;
    for (size_t i = 0; i < n; ++i)
    {
        float const gpu_val = static_cast<float>(gpu[i]);
        float const ref_val = ref[i];
        if (!std::isfinite(gpu_val) || !std::isfinite(ref_val))
        {
            bool const both_nan = std::isnan(gpu_val) && std::isnan(ref_val);
            bool const both_inf = std::isinf(gpu_val) && std::isinf(ref_val)
                && (std::signbit(gpu_val) == std::signbit(ref_val));
            if (both_nan || both_inf)
            {
                continue;
            }
            stats.max_abs = std::numeric_limits<float>::infinity();
            stats.max_rel = std::numeric_limits<float>::infinity();
            stats.max_idx = i;
            return stats;
        }
        float const abs_err = std::fabs(gpu_val - ref_val);
        float const rel_err = abs_err / (std::fabs(ref_val) + 1e-6f);
        if (abs_err > stats.max_abs)
        {
            stats.max_abs = abs_err;
            stats.max_idx = i;
        }
        if (rel_err > stats.max_rel)
        {
            stats.max_rel = rel_err;
        }
    }
    return stats;
}

void fc1_reference(std::vector<float>& out_fp32, std::vector<cutlass::half_t>& out_half,
    std::vector<cutlass::half_t> const& input, std::vector<cutlass::half_t> const& weights,
    std::vector<int64_t> const& total_tokens_including_expert, int num_experts, int64_t num_rows, int64_t gemm_n,
    int64_t gemm_k)
{
    out_fp32.assign(static_cast<size_t>(num_rows * gemm_n), 0.0f);
    out_half.assign(static_cast<size_t>(num_rows * gemm_n), cutlass::half_t(0.0f));
    for (int e = 0; e < num_experts; ++e)
    {
        int64_t const start = (e == 0) ? 0 : total_tokens_including_expert[e - 1];
        int64_t const end = total_tokens_including_expert[e];
        size_t const expert_stride = static_cast<size_t>(2 * gemm_n * gemm_k);
        cutlass::half_t const* w_value = weights.data() + static_cast<size_t>(e) * expert_stride;
        cutlass::half_t const* w_gate = w_value + static_cast<size_t>(gemm_n * gemm_k);
        for (int64_t row = start; row < end; ++row)
        {
            cutlass::half_t const* a = input.data() + static_cast<size_t>(row) * gemm_k;
            for (int64_t n = 0; n < gemm_n; ++n)
            {
                float acc_value = 0.0f;
                float acc_gate = 0.0f;
                for (int64_t k = 0; k < gemm_k; ++k)
                {
                    float const a_val = static_cast<float>(a[k]);
                    acc_value += a_val * static_cast<float>(w_value[static_cast<size_t>(n) * gemm_k + k]);
                    acc_gate += a_val * static_cast<float>(w_gate[static_cast<size_t>(n) * gemm_k + k]);
                }
                float const out = silu(acc_gate) * acc_value;
                size_t const idx = static_cast<size_t>(row) * gemm_n + static_cast<size_t>(n);
                out_fp32[idx] = out;
                out_half[idx] = cutlass::half_t(out);
            }
        }
    }
}

void fc2_reference(std::vector<float>& out_fp32, std::vector<cutlass::half_t> const& input,
    std::vector<cutlass::half_t> const& weights, std::vector<int64_t> const& total_tokens_including_expert,
    int num_experts, int64_t num_rows, int64_t gemm_n, int64_t gemm_k)
{
    out_fp32.assign(static_cast<size_t>(num_rows * gemm_n), 0.0f);
    for (int e = 0; e < num_experts; ++e)
    {
        int64_t const start = (e == 0) ? 0 : total_tokens_including_expert[e - 1];
        int64_t const end = total_tokens_including_expert[e];
        size_t const expert_stride = static_cast<size_t>(gemm_n * gemm_k);
        cutlass::half_t const* w = weights.data() + static_cast<size_t>(e) * expert_stride;
        for (int64_t row = start; row < end; ++row)
        {
            cutlass::half_t const* a = input.data() + static_cast<size_t>(row) * gemm_k;
            for (int64_t n = 0; n < gemm_n; ++n)
            {
                float acc = 0.0f;
                for (int64_t k = 0; k < gemm_k; ++k)
                {
                    acc += static_cast<float>(a[k]) * static_cast<float>(w[static_cast<size_t>(n) * gemm_k + k]);
                }
                out_fp32[static_cast<size_t>(row) * gemm_n + static_cast<size_t>(n)] = acc;
            }
        }
    }
}
} // namespace

int main(int argc, char** argv)
{
    Args args;
    parse_args(argc, argv, args);

    if (args.list_configs)
    {
        print_all_configs();
        return 0;
    }

    if (args.num_tokens <= 0 || args.hidden_size <= 0 || args.inter_size <= 0 || args.num_experts <= 0
        || args.experts_per_token <= 0)
    {
        std::fprintf(stderr, "All sizes must be positive.\n");
        return 1;
    }
    if (args.experts_per_token > args.num_experts)
    {
        std::fprintf(stderr, "experts_per_token must be <= num_experts.\n");
        return 1;
    }
    if ((args.hidden_size % 8) != 0 || (args.inter_size % 8) != 0)
    {
        std::fprintf(stderr, "hidden_size and inter_size must be multiples of 8.\n");
        return 1;
    }
    if ((args.hidden_size % 64) != 0 || (args.inter_size % 64) != 0)
    {
        std::fprintf(stderr, "hidden_size and inter_size must be multiples of 64 for fused configs.\n");
        return 1;
    }

    int64_t const num_rows = args.num_tokens * static_cast<int64_t>(args.experts_per_token);
    if (num_rows <= 0)
    {
        std::fprintf(stderr, "num_rows overflowed.\n");
        return 1;
    }

    std::printf("num_tokens=%ld hidden_size=%ld inter_size=%ld num_experts=%d experts_per_token=%d\n",
        args.num_tokens, args.hidden_size, args.inter_size, args.num_experts, args.experts_per_token);
    std::printf("num_rows=%ld (expanded)\n", num_rows);

    size_t const input_elems = checked_mul_size(static_cast<size_t>(num_rows), static_cast<size_t>(args.hidden_size),
        "expanded_input");
    size_t const fc1_weight_elems = checked_mul_size(
        checked_mul_size(static_cast<size_t>(args.num_experts) * 2ULL, static_cast<size_t>(args.inter_size),
            "fc1_weight"),
        static_cast<size_t>(args.hidden_size), "fc1_weight");
    size_t const fc2_weight_elems = checked_mul_size(
        checked_mul_size(static_cast<size_t>(args.num_experts), static_cast<size_t>(args.hidden_size), "fc2_weight"),
        static_cast<size_t>(args.inter_size), "fc2_weight");
    size_t const fc1_output_elems
        = checked_mul_size(static_cast<size_t>(num_rows), static_cast<size_t>(args.inter_size), "fc1_output");
    size_t const fc2_output_elems
        = checked_mul_size(static_cast<size_t>(num_rows), static_cast<size_t>(args.hidden_size), "fc2_output");

    bool do_verify = false;
    if (args.verify)
    {
        bool const small_case = args.num_tokens <= 64 && args.hidden_size <= 256 && args.inter_size <= 256
            && args.num_experts <= 8 && args.experts_per_token <= 4;
        if (small_case)
        {
            do_verify = true;
        }
        else
        {
            std::printf("verify requested, but sizes are too large; skipping CPU reference.\n");
        }
    }

    float const fc1_weight_scale = 0.01f;
    float const fc2_weight_scale = do_verify ? 0.002f : 0.02f;
    float const fc2_input_scale = do_verify ? 0.003f : 0.03f;

    std::vector<cutlass::half_t> h_input(static_cast<size_t>(args.num_tokens) * args.hidden_size);
    for (size_t i = 0; i < h_input.size(); ++i)
    {
        h_input[i] = cutlass::half_t(static_cast<float>(i % 101) * 0.01f);
    }

    std::vector<int64_t> expert_counts(static_cast<size_t>(args.num_experts), 0);
    for (int64_t t = 0; t < args.num_tokens; ++t)
    {
        for (int i = 0; i < args.experts_per_token; ++i)
        {
            int const expert = static_cast<int>((t + i) % args.num_experts);
            expert_counts[expert] += 1;
        }
    }

    std::vector<int64_t> total_tokens_including_expert(static_cast<size_t>(args.num_experts), 0);
    int64_t running = 0;
    for (int e = 0; e < args.num_experts; ++e)
    {
        running += expert_counts[e];
        total_tokens_including_expert[e] = running;
    }

    std::vector<int64_t> write_offsets(static_cast<size_t>(args.num_experts), 0);
    for (int e = 1; e < args.num_experts; ++e)
    {
        write_offsets[e] = total_tokens_including_expert[e - 1];
    }

    std::vector<cutlass::half_t> h_expanded_input(input_elems);
    for (int64_t t = 0; t < args.num_tokens; ++t)
    {
        cutlass::half_t const* src = h_input.data() + static_cast<size_t>(t) * args.hidden_size;
        for (int i = 0; i < args.experts_per_token; ++i)
        {
            int const expert = static_cast<int>((t + i) % args.num_experts);
            int64_t const row = write_offsets[expert]++;
            cutlass::half_t* dst = h_expanded_input.data() + static_cast<size_t>(row) * args.hidden_size;
            std::memcpy(dst, src, sizeof(cutlass::half_t) * static_cast<size_t>(args.hidden_size));
        }
    }
    for (int e = 0; e < args.num_experts; ++e)
    {
        if (write_offsets[e] != total_tokens_including_expert[e])
        {
            std::fprintf(stderr, "Mismatch in expert row counts for expert %d.\n", e);
            return 1;
        }
    }

    cudaStream_t stream{};
    tensorrt_llm::common::check_cuda_error(cudaStreamCreate(&stream));

    cutlass::half_t* d_input = nullptr;
    tensorrt_llm::common::check_cuda_error(
        cudaMalloc(&d_input, sizeof(cutlass::half_t) * h_expanded_input.size()));
    tensorrt_llm::common::check_cuda_error(cudaMemcpy(
        d_input, h_expanded_input.data(), sizeof(cutlass::half_t) * h_expanded_input.size(), cudaMemcpyHostToDevice));

    int64_t* d_total_tokens = nullptr;
    tensorrt_llm::common::check_cuda_error(
        cudaMalloc(&d_total_tokens, sizeof(int64_t) * total_tokens_including_expert.size()));
    tensorrt_llm::common::check_cuda_error(
        cudaMemcpy(d_total_tokens, total_tokens_including_expert.data(),
            sizeof(int64_t) * total_tokens_including_expert.size(), cudaMemcpyHostToDevice));

    cutlass::half_t* d_fc1_weight = nullptr;
    cutlass::half_t* d_fc2_weight = nullptr;
    std::vector<cutlass::half_t> h_fc1_weight;
    std::vector<cutlass::half_t> h_fc2_weight;
    if (args.run_fc1)
    {
        tensorrt_llm::common::check_cuda_error(
            cudaMalloc(&d_fc1_weight, sizeof(cutlass::half_t) * fc1_weight_elems));
        if (do_verify)
        {
            h_fc1_weight.resize(fc1_weight_elems);
            fill_host_half(h_fc1_weight.data(), fc1_weight_elems, fc1_weight_scale);
            tensorrt_llm::common::check_cuda_error(
                cudaMemcpy(d_fc1_weight, h_fc1_weight.data(), sizeof(cutlass::half_t) * fc1_weight_elems,
                    cudaMemcpyHostToDevice));
        }
        else
        {
            fill_device_half(d_fc1_weight, fc1_weight_elems, fc1_weight_scale, stream);
        }
    }
    if (args.run_fc2)
    {
        tensorrt_llm::common::check_cuda_error(
            cudaMalloc(&d_fc2_weight, sizeof(cutlass::half_t) * fc2_weight_elems));
        if (do_verify)
        {
            h_fc2_weight.resize(fc2_weight_elems);
            fill_host_half(h_fc2_weight.data(), fc2_weight_elems, fc2_weight_scale);
            tensorrt_llm::common::check_cuda_error(
                cudaMemcpy(d_fc2_weight, h_fc2_weight.data(), sizeof(cutlass::half_t) * fc2_weight_elems,
                    cudaMemcpyHostToDevice));
        }
        else
        {
            fill_device_half(d_fc2_weight, fc2_weight_elems, fc2_weight_scale, stream);
        }
    }

    cutlass::half_t* d_fc1_output = nullptr;
    if (args.run_fc1)
    {
        tensorrt_llm::common::check_cuda_error(
            cudaMalloc(&d_fc1_output, sizeof(cutlass::half_t) * fc1_output_elems));
    }

    cutlass::half_t* d_fc2_input = nullptr;
    std::vector<cutlass::half_t> h_fc2_input;
    if (args.run_fc2)
    {
        if (args.run_fc1)
        {
            d_fc2_input = d_fc1_output;
        }
        else
        {
            tensorrt_llm::common::check_cuda_error(
                cudaMalloc(&d_fc2_input, sizeof(cutlass::half_t) * fc1_output_elems));
            if (do_verify)
            {
                h_fc2_input.resize(fc1_output_elems);
                fill_host_half(h_fc2_input.data(), fc1_output_elems, fc2_input_scale);
                tensorrt_llm::common::check_cuda_error(
                    cudaMemcpy(d_fc2_input, h_fc2_input.data(), sizeof(cutlass::half_t) * fc1_output_elems,
                        cudaMemcpyHostToDevice));
            }
            else
            {
                fill_device_half(d_fc2_input, fc1_output_elems, fc2_input_scale, stream);
            }
        }
    }

    cutlass::half_t* d_fc2_output = nullptr;
    if (args.run_fc2)
    {
        tensorrt_llm::common::check_cuda_error(
            cudaMalloc(&d_fc2_output, sizeof(cutlass::half_t) * fc2_output_elems));
    }

    int device = 0;
    int sm_count = 0;
    tensorrt_llm::common::check_cuda_error(cudaGetDevice(&device));
    tensorrt_llm::common::check_cuda_error(
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));

    if (args.force_config)
    {
        if (!tensorrt_llm::kernels::cutlass_kernels_oss::sm80_fused_moe_is_supported_config(args.forced_config))
        {
            std::fprintf(stderr, "Requested config is not supported. Use --list_configs to see valid options.\n");
            return 1;
        }
        std::printf("forcing config: tile_m=%d tile_n=%d tile_k=%d stages=%d\n", args.forced_config.tile_m,
            args.forced_config.tile_n, args.forced_config.tile_k, args.forced_config.stages);
    }

    tensorrt_llm::kernels::cutlass_kernels_oss::Sm80FusedMoeGemmConfig fc1_cfg{};
    tensorrt_llm::kernels::cutlass_kernels_oss::Sm80FusedMoeGemmConfig fc2_cfg{};
    if (args.run_fc1)
    {
        if (args.force_config)
        {
            fc1_cfg = args.forced_config;
        }
        else
        {
            bool ok = tensorrt_llm::kernels::cutlass_kernels_oss::sm80_fused_moe_select_config_fc1(fc1_cfg,
                total_tokens_including_expert.data(), args.num_experts, num_rows, args.inter_size, args.hidden_size,
                sm_count);
            if (!ok)
            {
                std::fprintf(stderr, "No valid fused config found for FC1.\n");
                return 1;
            }
        }
        std::printf("fc1 config: tile_m=%d tile_n=%d tile_k=%d stages=%d\n", fc1_cfg.tile_m, fc1_cfg.tile_n,
            fc1_cfg.tile_k, fc1_cfg.stages);
    }

    if (args.run_fc2)
    {
        if (args.force_config)
        {
            fc2_cfg = args.forced_config;
        }
        else
        {
            bool ok = tensorrt_llm::kernels::cutlass_kernels_oss::sm80_fused_moe_select_config_fc2(fc2_cfg,
                total_tokens_including_expert.data(), args.num_experts, num_rows, args.hidden_size, args.inter_size,
                sm_count);
            if (!ok)
            {
                std::fprintf(stderr, "No valid fused config found for FC2.\n");
                return 1;
            }
        }
        std::printf("fc2 config: tile_m=%d tile_n=%d tile_k=%d stages=%d\n", fc2_cfg.tile_m, fc2_cfg.tile_n,
            fc2_cfg.tile_k, fc2_cfg.stages);
    }

    if (args.run_fc1)
    {
        tensorrt_llm::kernels::cutlass_kernels_oss::sm80_fused_moe_fc1_swiglu_fp16_with_config(d_input, d_fc1_weight,
            nullptr, true, d_fc1_output, d_total_tokens, num_rows, args.inter_size, args.hidden_size, args.num_experts,
            sm_count, stream, nullptr, fc1_cfg);
    }

    if (args.run_fc2)
    {
        tensorrt_llm::kernels::cutlass_kernels_oss::sm80_fused_moe_fc2_identity_fp16_with_config(d_fc2_input,
            d_fc2_weight, nullptr, true, d_fc2_output, d_total_tokens, num_rows, args.hidden_size, args.inter_size,
            args.num_experts, sm_count, stream, nullptr, fc2_cfg);
    }

    tensorrt_llm::common::check_cuda_error(cudaStreamSynchronize(stream));

    std::vector<cutlass::half_t> h_fc1_out;
    std::vector<cutlass::half_t> h_fc2_out;

    if (args.run_fc1)
    {
        size_t const copy_elems = do_verify ? fc1_output_elems : std::min<size_t>(fc1_output_elems, 1024);
        h_fc1_out.resize(copy_elems);
        tensorrt_llm::common::check_cuda_error(cudaMemcpy(
            h_fc1_out.data(), d_fc1_output, sizeof(cutlass::half_t) * h_fc1_out.size(), cudaMemcpyDeviceToHost));
        if (!do_verify)
        {
            std::printf("fc1 checksum=%.6f\n", checksum_first(h_fc1_out.data(), h_fc1_out.size()));
        }
    }
    if (args.run_fc2)
    {
        size_t const copy_elems = do_verify ? fc2_output_elems : std::min<size_t>(fc2_output_elems, 1024);
        h_fc2_out.resize(copy_elems);
        tensorrt_llm::common::check_cuda_error(cudaMemcpy(
            h_fc2_out.data(), d_fc2_output, sizeof(cutlass::half_t) * h_fc2_out.size(), cudaMemcpyDeviceToHost));
        if (!do_verify)
        {
            std::printf("fc2 checksum=%.6f\n", checksum_first(h_fc2_out.data(), h_fc2_out.size()));
        }
    }

    if (do_verify)
    {
        std::vector<float> fc1_ref_fp32;
        std::vector<cutlass::half_t> fc1_ref_half;
        std::vector<float> fc2_ref_fp32;

        if (args.run_fc1)
        {
            fc1_reference(fc1_ref_fp32, fc1_ref_half, h_expanded_input, h_fc1_weight, total_tokens_including_expert,
                args.num_experts, num_rows, args.inter_size, args.hidden_size);

            std::vector<float> fc1_ref_fp16(fc1_ref_half.size());
            for (size_t i = 0; i < fc1_ref_half.size(); ++i)
            {
                fc1_ref_fp16[i] = static_cast<float>(fc1_ref_half[i]);
            }

            find_nonfinite_half(h_fc1_out.data(), h_fc1_out.size(), "fc1_gpu");
            find_nonfinite_float(fc1_ref_fp16.data(), fc1_ref_fp16.size(), "fc1_ref");
            DiffStats stats = compare_half_to_float(h_fc1_out.data(), fc1_ref_fp16.data(), h_fc1_out.size());
            float const abs_tol = 5e-2f;
            float const rel_tol = 5e-2f;
            bool const pass = (stats.max_abs <= abs_tol) || (stats.max_rel <= rel_tol);
            std::printf("fc1 verify: max_abs=%.6f max_rel=%.6f %s\n", stats.max_abs, stats.max_rel,
                pass ? "PASS" : "FAIL");
        }

        if (args.run_fc2)
        {
            std::vector<cutlass::half_t> const& fc2_input = args.run_fc1 ? h_fc1_out : h_fc2_input;
            fc2_reference(fc2_ref_fp32, fc2_input, h_fc2_weight, total_tokens_including_expert, args.num_experts,
                num_rows, args.hidden_size, args.inter_size);

            std::vector<float> fc2_ref_fp16(fc2_ref_fp32.size());
            for (size_t i = 0; i < fc2_ref_fp32.size(); ++i)
            {
                cutlass::half_t hval = cutlass::half_t(fc2_ref_fp32[i]);
                fc2_ref_fp16[i] = static_cast<float>(hval);
            }

            find_nonfinite_half(h_fc2_out.data(), h_fc2_out.size(), "fc2_gpu");
            find_nonfinite_float(fc2_ref_fp16.data(), fc2_ref_fp16.size(), "fc2_ref");
            DiffStats stats = compare_half_to_float(h_fc2_out.data(), fc2_ref_fp16.data(), h_fc2_out.size());
            float const abs_tol = 5e-2f;
            float const rel_tol = 5e-2f;
            bool const pass = (stats.max_abs <= abs_tol) || (stats.max_rel <= rel_tol);
            std::printf("fc2 verify: max_abs=%.6f max_rel=%.6f %s\n", stats.max_abs, stats.max_rel,
                pass ? "PASS" : "FAIL");
        }
    }

    tensorrt_llm::common::check_cuda_error(cudaFree(d_input));
    tensorrt_llm::common::check_cuda_error(cudaFree(d_total_tokens));
    if (d_fc1_weight)
    {
        tensorrt_llm::common::check_cuda_error(cudaFree(d_fc1_weight));
    }
    if (d_fc2_weight)
    {
        tensorrt_llm::common::check_cuda_error(cudaFree(d_fc2_weight));
    }
    if (d_fc1_output)
    {
        tensorrt_llm::common::check_cuda_error(cudaFree(d_fc1_output));
    }
    if (args.run_fc2 && !args.run_fc1 && d_fc2_input)
    {
        tensorrt_llm::common::check_cuda_error(cudaFree(d_fc2_input));
    }
    if (d_fc2_output)
    {
        tensorrt_llm::common::check_cuda_error(cudaFree(d_fc2_output));
    }
    tensorrt_llm::common::check_cuda_error(cudaStreamDestroy(stream));

    return 0;
}
