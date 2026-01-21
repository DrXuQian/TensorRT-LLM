/*
 * MoE SwiGLU Activation Kernel - Standalone Version
 *
 * Extracted from TensorRT-LLM moe_kernels.cu
 * This kernel applies gated activation (SwiGLU/GeGLU) to FC1 output.
 *
 * SwiGLU: output = silu(gate) * linear
 * GeGLU:  output = gelu(gate) * linear
 *
 * Original function: doGatedActivationKernel
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cassert>
#include <cmath>

// ============================================================================
// Tunable Parameters
// ============================================================================
// ACTIVATION_THREADS_PER_BLOCK: Number of threads per block (default: 256)
//   - Each thread processes 128 bits (8 FP16 elements) per iteration
//   - Block size affects occupancy
//
// ACTIVATION_ELEM_PER_THREAD: Elements per thread
//   - For FP16: 128 bits / 16 bits = 8 elements
//   - Must be power of 2 for vectorized loads
//
// Activation Types:
//   - SwiGLU: SiLU(gate) * linear
//   - GeGLU: GELU(gate) * linear
//   - SwiGLUBias: SwiGLU with alpha, beta, limit parameters per expert
// ============================================================================

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

constexpr int ACTIVATION_THREADS_PER_BLOCK = 256;

// Activation type enum
enum class ActivationType {
    Swiglu,
    Geglu,
    SwigluBias
};

// ============================================================================
// Activation Functions
// ============================================================================

// SiLU (Swish) activation: x * sigmoid(x)
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

// GELU activation (approximate)
__device__ __forceinline__ float gelu(float x) {
    const float kSqrt2OverPi = 0.7978845608028654f;
    const float kCoeff = 0.044715f;
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(kSqrt2OverPi * (x + kCoeff * x3)));
}

// Helper: find which expert a token belongs to
__device__ __forceinline__ int findExpertForToken(
    int64_t const* expert_first_token_offset,
    int num_experts,
    int64_t token)
{
    int lo = 0, hi = num_experts;
    while (lo < hi) {
        int mid = (lo + hi + 1) / 2;
        if (expert_first_token_offset[mid] <= token) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return lo;
}

// ============================================================================
// Gated Activation Kernel
// ============================================================================
// Input layout: [num_tokens, inter_size * 2]
//   - First inter_size columns: linear values
//   - Second inter_size columns: gate values
// Output layout: [num_tokens, inter_size]
template <ActivationType ACT_TYPE>
__global__ void doGatedActivationKernel(
    half* output,
    half const* gemm_result,
    int64_t const* expert_first_token_offset,
    int64_t inter_size,
    int64_t num_experts_per_node,
    // Optional per-expert parameters for SwigluBias
    float const* swiglu_alpha = nullptr,
    float const* swiglu_beta = nullptr,
    float const* swiglu_limit = nullptr)
{
    int64_t const tid = threadIdx.x;
    int64_t const token = blockIdx.x;

    // Check if token is valid
    if (token >= expert_first_token_offset[num_experts_per_node]) {
        return;
    }

    // Point to this token's data
    half* output_ptr = output + token * inter_size;
    half const* gemm_ptr = gemm_result + token * inter_size * 2;

    // Each thread processes 8 FP16 elements (128 bits)
    constexpr int64_t ELEM_PER_THREAD = 8;

    int64_t const start_offset = tid;
    int64_t const stride = ACTIVATION_THREADS_PER_BLOCK;
    int64_t const num_elems_in_col = inter_size / ELEM_PER_THREAD;
    int64_t const inter_size_vec = inter_size / ELEM_PER_THREAD;

    // Get per-expert parameters if using SwigluBias
    float alpha = 1.0f;
    float beta = 0.0f;
    float limit = INFINITY;

    if constexpr (ACT_TYPE == ActivationType::SwigluBias) {
        int expert = findExpertForToken(expert_first_token_offset, num_experts_per_node, token + 1) - 1;
        alpha = swiglu_alpha ? swiglu_alpha[expert] : 1.0f;
        beta = swiglu_beta ? swiglu_beta[expert] : 0.0f;
        limit = swiglu_limit ? swiglu_limit[expert] : INFINITY;
    }

    // Process elements
    for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
        // Load 8 elements for linear and gate
        float4 linear_vec = *reinterpret_cast<float4 const*>(
            reinterpret_cast<half const*>(gemm_ptr) + elem_index * ELEM_PER_THREAD);
        float4 gate_vec = *reinterpret_cast<float4 const*>(
            reinterpret_cast<half const*>(gemm_ptr) + (elem_index + inter_size_vec) * ELEM_PER_THREAD);

        // Convert to float arrays for computation
        half* linear_half = reinterpret_cast<half*>(&linear_vec);
        half* gate_half = reinterpret_cast<half*>(&gate_vec);
        float output_vals[ELEM_PER_THREAD];

        #pragma unroll
        for (int i = 0; i < ELEM_PER_THREAD; i++) {
            float linear_val = __half2float(linear_half[i]);
            float gate_val = __half2float(gate_half[i]);

            float gate_act;
            if constexpr (ACT_TYPE == ActivationType::Geglu) {
                gate_act = gelu(gate_val);
            } else {
                // SwiGLU or SwigluBias
                gate_act = silu(gate_val);
            }

            // Apply per-expert scaling for SwigluBias
            if constexpr (ACT_TYPE == ActivationType::SwigluBias) {
                gate_act = alpha * gate_act + beta;
                gate_act = fminf(fmaxf(gate_act, -limit), limit);
            }

            output_vals[i] = gate_act * linear_val;
        }

        // Convert back to half and store
        half output_half[ELEM_PER_THREAD];
        #pragma unroll
        for (int i = 0; i < ELEM_PER_THREAD; i++) {
            output_half[i] = __float2half(output_vals[i]);
        }

        *reinterpret_cast<float4*>(output_ptr + elem_index * ELEM_PER_THREAD) =
            *reinterpret_cast<float4*>(output_half);
    }
}

// ============================================================================
// Launcher function
// ============================================================================
void launchSwiGLUKernel(
    half* output,
    half const* gemm_result,
    int64_t const* expert_first_token_offset,
    int64_t inter_size,
    int64_t num_tokens,
    int64_t num_experts,
    ActivationType activation_type,
    float const* swiglu_alpha,
    float const* swiglu_beta,
    float const* swiglu_limit,
    cudaStream_t stream)
{
    int64_t const blocks = num_tokens;
    int64_t const threads = ACTIVATION_THREADS_PER_BLOCK;

    switch (activation_type) {
        case ActivationType::Swiglu:
            doGatedActivationKernel<ActivationType::Swiglu><<<blocks, threads, 0, stream>>>(
                output, gemm_result, expert_first_token_offset, inter_size, num_experts);
            break;
        case ActivationType::Geglu:
            doGatedActivationKernel<ActivationType::Geglu><<<blocks, threads, 0, stream>>>(
                output, gemm_result, expert_first_token_offset, inter_size, num_experts);
            break;
        case ActivationType::SwigluBias:
            doGatedActivationKernel<ActivationType::SwigluBias><<<blocks, threads, 0, stream>>>(
                output, gemm_result, expert_first_token_offset, inter_size, num_experts,
                swiglu_alpha, swiglu_beta, swiglu_limit);
            break;
    }
}

// ============================================================================
// Test program
// ============================================================================
int main() {
    std::cout << "=== MoE SwiGLU Activation Kernel Test ===" << std::endl;

    // Test configuration
    constexpr int NUM_TOKENS = 1024;  // Expanded tokens
    constexpr int INTER_SIZE = 768;   // Intermediate size (FC1 output / 2)
    constexpr int NUM_EXPERTS = 128;

    std::cout << "Configuration:" << std::endl;
    std::cout << "  num_tokens: " << NUM_TOKENS << std::endl;
    std::cout << "  inter_size: " << INTER_SIZE << std::endl;
    std::cout << "  num_experts: " << NUM_EXPERTS << std::endl;

    // Allocate host memory
    // FC1 output: [num_tokens, inter_size * 2]
    std::vector<half> h_fc1_output(NUM_TOKENS * INTER_SIZE * 2);
    std::vector<half> h_activation_output(NUM_TOKENS * INTER_SIZE);
    std::vector<int64_t> h_expert_offset(NUM_EXPERTS + 1);

    // Initialize FC1 output with random values
    srand(42);
    for (int i = 0; i < NUM_TOKENS * INTER_SIZE * 2; i++) {
        h_fc1_output[i] = __float2half((float)(rand() % 200 - 100) / 100.0f);
    }

    // Create uniform expert distribution
    int tokens_per_expert = NUM_TOKENS / NUM_EXPERTS;
    for (int i = 0; i <= NUM_EXPERTS; i++) {
        h_expert_offset[i] = i * tokens_per_expert;
    }

    // Allocate device memory
    half *d_fc1_output, *d_activation_output;
    int64_t *d_expert_offset;

    CUDA_CHECK(cudaMalloc(&d_fc1_output, NUM_TOKENS * INTER_SIZE * 2 * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_activation_output, NUM_TOKENS * INTER_SIZE * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_expert_offset, (NUM_EXPERTS + 1) * sizeof(int64_t)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_fc1_output, h_fc1_output.data(),
        NUM_TOKENS * INTER_SIZE * 2 * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_expert_offset, h_expert_offset.data(),
        (NUM_EXPERTS + 1) * sizeof(int64_t), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Warm-up
    for (int i = 0; i < 3; i++) {
        launchSwiGLUKernel(d_activation_output, d_fc1_output, d_expert_offset,
            INTER_SIZE, NUM_TOKENS, NUM_EXPERTS, ActivationType::Swiglu,
            nullptr, nullptr, nullptr, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Benchmark
    constexpr int ITERS = 1000;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; i++) {
        launchSwiGLUKernel(d_activation_output, d_fc1_output, d_expert_offset,
            INTER_SIZE, NUM_TOKENS, NUM_EXPERTS, ActivationType::Swiglu,
            nullptr, nullptr, nullptr, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();

    double avg_us = std::chrono::duration<double, std::micro>(end - start).count() / ITERS;
    double bandwidth_gbps = (double)(NUM_TOKENS * INTER_SIZE * 2 * sizeof(half) +
                                     NUM_TOKENS * INTER_SIZE * sizeof(half)) / avg_us / 1000.0;

    std::cout << "\nPerformance:" << std::endl;
    std::cout << "  Average time: " << avg_us << " us" << std::endl;
    std::cout << "  Effective bandwidth: " << bandwidth_gbps << " GB/s" << std::endl;

    // Verify results (CPU reference)
    CUDA_CHECK(cudaMemcpy(h_activation_output.data(), d_activation_output,
        NUM_TOKENS * INTER_SIZE * sizeof(half), cudaMemcpyDeviceToHost));

    bool correct = true;
    for (int token = 0; token < std::min(10, NUM_TOKENS); token++) {
        for (int col = 0; col < 4; col++) {
            float linear = __half2float(h_fc1_output[token * INTER_SIZE * 2 + col]);
            float gate = __half2float(h_fc1_output[token * INTER_SIZE * 2 + INTER_SIZE + col]);
            float expected = linear * (gate / (1.0f + expf(-gate)));  // SiLU(gate) * linear
            float actual = __half2float(h_activation_output[token * INTER_SIZE + col]);

            if (fabsf(expected - actual) > 0.01f) {
                std::cout << "Mismatch at token " << token << " col " << col
                          << ": expected " << expected << ", got " << actual << std::endl;
                correct = false;
            }
        }
    }

    std::cout << "\nVerification: " << (correct ? "PASSED" : "FAILED") << std::endl;

    // Cleanup
    CUDA_CHECK(cudaFree(d_fc1_output));
    CUDA_CHECK(cudaFree(d_activation_output));
    CUDA_CHECK(cudaFree(d_expert_offset));
    CUDA_CHECK(cudaStreamDestroy(stream));

    std::cout << "\nTest completed!" << std::endl;
    return 0;
}
