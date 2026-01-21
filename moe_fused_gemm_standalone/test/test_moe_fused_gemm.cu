/*
 * Standalone test for SM80 fused MoE GEMM kernels (FP16).
 */

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/fused_moe_gemm_sm80_wrappers.h"
#include <algorithm>
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
};

void print_usage(char const* name)
{
    std::printf(
        "Usage: %s [--num_tokens=N] [--hidden_size=N] [--inter_size=N] [--num_experts=N] "
        "[--experts_per_token=N] [--op=fc1|fc2|both]\n",
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
} // namespace

int main(int argc, char** argv)
{
    Args args;
    parse_args(argc, argv, args);

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
    if (args.run_fc1)
    {
        tensorrt_llm::common::check_cuda_error(
            cudaMalloc(&d_fc1_weight, sizeof(cutlass::half_t) * fc1_weight_elems));
        fill_device_half(d_fc1_weight, fc1_weight_elems, 0.01f, stream);
    }
    if (args.run_fc2)
    {
        tensorrt_llm::common::check_cuda_error(
            cudaMalloc(&d_fc2_weight, sizeof(cutlass::half_t) * fc2_weight_elems));
        fill_device_half(d_fc2_weight, fc2_weight_elems, 0.02f, stream);
    }

    cutlass::half_t* d_fc1_output = nullptr;
    if (args.run_fc1)
    {
        tensorrt_llm::common::check_cuda_error(
            cudaMalloc(&d_fc1_output, sizeof(cutlass::half_t) * fc1_output_elems));
    }

    cutlass::half_t* d_fc2_input = nullptr;
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
            fill_device_half(d_fc2_input, fc1_output_elems, 0.03f, stream);
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

    if (args.run_fc1)
    {
        tensorrt_llm::kernels::cutlass_kernels_oss::sm80_fused_moe_fc1_swiglu_fp16(d_input, d_fc1_weight, nullptr,
            true, d_fc1_output, d_total_tokens, num_rows, args.inter_size, args.hidden_size, args.num_experts, sm_count,
            stream, nullptr);
    }

    if (args.run_fc2)
    {
        tensorrt_llm::kernels::cutlass_kernels_oss::sm80_fused_moe_fc2_identity_fp16(d_fc2_input, d_fc2_weight, nullptr,
            true, d_fc2_output, d_total_tokens, num_rows, args.hidden_size, args.inter_size, args.num_experts, sm_count,
            stream, nullptr);
    }

    tensorrt_llm::common::check_cuda_error(cudaStreamSynchronize(stream));

    if (args.run_fc1)
    {
        std::vector<cutlass::half_t> h_fc1_out(std::min<size_t>(fc1_output_elems, 1024));
        tensorrt_llm::common::check_cuda_error(cudaMemcpy(
            h_fc1_out.data(), d_fc1_output, sizeof(cutlass::half_t) * h_fc1_out.size(), cudaMemcpyDeviceToHost));
        std::printf("fc1 checksum=%.6f\n", checksum_first(h_fc1_out.data(), h_fc1_out.size()));
    }
    if (args.run_fc2)
    {
        std::vector<cutlass::half_t> h_fc2_out(std::min<size_t>(fc2_output_elems, 1024));
        tensorrt_llm::common::check_cuda_error(cudaMemcpy(
            h_fc2_out.data(), d_fc2_output, sizeof(cutlass::half_t) * h_fc2_out.size(), cudaMemcpyDeviceToHost));
        std::printf("fc2 checksum=%.6f\n", checksum_first(h_fc2_out.data(), h_fc2_out.size()));
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
