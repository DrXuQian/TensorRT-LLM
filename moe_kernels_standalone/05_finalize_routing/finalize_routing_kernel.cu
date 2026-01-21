/*
 * MoE Finalize Routing Kernel - Standalone Version
 *
 * Extracted from TensorRT-LLM moe_kernels.cu
 * This kernel unpermutes the MoE output and performs weighted reduction
 * to produce the final output.
 *
 * Original function: finalizeMoeRoutingKernel
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cassert>

// ============================================================================
// Tunable Parameters
// ============================================================================
// FINALIZE_THREADS_PER_BLOCK: Number of threads per block (default: 256)
//   - Each thread processes 128 bits per iteration
//   - Each block handles one output token
//
// FINALIZE_ELEM_PER_THREAD: Elements per thread
//   - For FP16: 128 bits / 16 bits = 8 elements
//   - For FP32: 128 bits / 32 bits = 4 elements
// ============================================================================

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

constexpr int FINALIZE_THREADS_PER_BLOCK = 256;

// Scale mode for routing weights
enum class ScaleMode {
    DEFAULT,    // Use routing weights
    NO_SCALE    // All weights = 1.0
};

// ============================================================================
// Finalize MoE Routing Kernel
// ============================================================================
// This kernel:
// 1. Unpermutes the expert outputs back to original token order
// 2. Performs weighted reduction across experts_per_token outputs
// 3. Optionally adds bias
template <typename OutputType, typename GemmOutputType, ScaleMode SCALE_MODE>
__global__ void finalizeMoeRoutingKernel(
    GemmOutputType const* expanded_permuted_rows,  // [num_tokens * k, hidden_size]
    OutputType* reduced_unpermuted_output,          // [num_tokens, hidden_size]
    GemmOutputType const* bias,                     // [num_experts, hidden_size] or nullptr
    float const* scales,                            // [num_tokens, k] routing weights
    int const* unpermuted_row_to_permuted_row,      // [num_tokens * k]
    int const* token_selected_experts,              // [num_tokens, k] expert indices
    int64_t const hidden_size,
    int64_t const experts_per_token,
    int const num_experts_per_node,
    int const start_expert_id = 0)
{
    assert(hidden_size % 4 == 0);

    int64_t const original_row = blockIdx.x;
    int64_t const num_rows = gridDim.x;
    OutputType* reduced_row_ptr = reduced_unpermuted_output + original_row * hidden_size;

    // Each thread processes 8 FP16 elements (128 bits)
    constexpr int64_t ELEM_PER_THREAD = 128 / (sizeof(GemmOutputType) * 8);

    int64_t const start_offset = threadIdx.x;
    int64_t const stride = FINALIZE_THREADS_PER_BLOCK;
    int64_t const num_elems_in_col = hidden_size / ELEM_PER_THREAD;

    // Process each column chunk
    for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
        // Accumulator for weighted sum
        float thread_output[ELEM_PER_THREAD];
        #pragma unroll
        for (int i = 0; i < ELEM_PER_THREAD; i++) {
            thread_output[i] = 0.0f;
        }

        // Sum contributions from all k selected experts
        for (int k_idx = 0; k_idx < experts_per_token; ++k_idx) {
            int64_t const k_offset = original_row * experts_per_token + k_idx;
            int64_t const expert_id = token_selected_experts[k_offset] - start_expert_id;

            // Skip if expert is not on this node
            if (expert_id < 0 || expert_id >= num_experts_per_node) {
                continue;
            }

            // Get the permuted row index for this expert's output
            int64_t const expanded_original_row = original_row + k_idx * num_rows;
            int64_t const expanded_permuted_row = unpermuted_row_to_permuted_row[expanded_original_row];

            // Get routing weight
            float const row_scale = (SCALE_MODE == ScaleMode::NO_SCALE) ? 1.0f : scales[k_offset];

            // Load expert output
            GemmOutputType const* expert_row_ptr = expanded_permuted_rows +
                expanded_permuted_row * hidden_size;

            // Load 128 bits
            float4 expert_data = *reinterpret_cast<float4 const*>(
                reinterpret_cast<GemmOutputType const*>(expert_row_ptr) + elem_index * ELEM_PER_THREAD);
            GemmOutputType* expert_vals = reinterpret_cast<GemmOutputType*>(&expert_data);

            // Add bias if provided
            if (bias) {
                float4 bias_data = *reinterpret_cast<float4 const*>(
                    reinterpret_cast<GemmOutputType const*>(bias + expert_id * hidden_size) +
                    elem_index * ELEM_PER_THREAD);
                GemmOutputType* bias_vals = reinterpret_cast<GemmOutputType*>(&bias_data);

                #pragma unroll
                for (int i = 0; i < ELEM_PER_THREAD; i++) {
                    float expert_val = __half2float(expert_vals[i]);
                    float bias_val = __half2float(bias_vals[i]);
                    thread_output[i] += row_scale * (expert_val + bias_val);
                }
            } else {
                #pragma unroll
                for (int i = 0; i < ELEM_PER_THREAD; i++) {
                    float expert_val = __half2float(expert_vals[i]);
                    thread_output[i] += row_scale * expert_val;
                }
            }
        }

        // Convert back to output type and store
        OutputType output_vals[ELEM_PER_THREAD];
        #pragma unroll
        for (int i = 0; i < ELEM_PER_THREAD; i++) {
            output_vals[i] = __float2half(thread_output[i]);
        }
        *reinterpret_cast<float4*>(reduced_row_ptr + elem_index * ELEM_PER_THREAD) =
            *reinterpret_cast<float4*>(output_vals);
    }
}

// ============================================================================
// Launcher function
// ============================================================================
template <typename T>
void launchFinalizeMoeRoutingKernel(
    T const* expanded_permuted_rows,
    T* reduced_unpermuted_output,
    T const* bias,
    float const* scales,
    int const* unpermuted_row_to_permuted_row,
    int const* token_selected_experts,
    int64_t num_tokens,
    int64_t hidden_size,
    int experts_per_token,
    int num_experts,
    bool use_scales,
    cudaStream_t stream)
{
    int64_t blocks = num_tokens;
    int64_t threads = FINALIZE_THREADS_PER_BLOCK;

    if (use_scales) {
        finalizeMoeRoutingKernel<T, T, ScaleMode::DEFAULT><<<blocks, threads, 0, stream>>>(
            expanded_permuted_rows, reduced_unpermuted_output, bias, scales,
            unpermuted_row_to_permuted_row, token_selected_experts,
            hidden_size, experts_per_token, num_experts, 0);
    } else {
        finalizeMoeRoutingKernel<T, T, ScaleMode::NO_SCALE><<<blocks, threads, 0, stream>>>(
            expanded_permuted_rows, reduced_unpermuted_output, bias, scales,
            unpermuted_row_to_permuted_row, token_selected_experts,
            hidden_size, experts_per_token, num_experts, 0);
    }
}

// Explicit instantiations
template void launchFinalizeMoeRoutingKernel<half>(
    half const*, half*, half const*, float const*, int const*, int const*,
    int64_t, int64_t, int, int, bool, cudaStream_t);

template void launchFinalizeMoeRoutingKernel<float>(
    float const*, float*, float const*, float const*, int const*, int const*,
    int64_t, int64_t, int, int, bool, cudaStream_t);

// ============================================================================
// Test program
// ============================================================================
int main() {
    std::cout << "=== MoE Finalize Routing Kernel Test ===" << std::endl;

    // Test configuration
    constexpr int NUM_TOKENS = 128;
    constexpr int HIDDEN_SIZE = 2048;
    constexpr int NUM_EXPERTS = 128;
    constexpr int EXPERTS_PER_TOKEN = 8;
    constexpr int EXPANDED_TOKENS = NUM_TOKENS * EXPERTS_PER_TOKEN;

    std::cout << "Configuration:" << std::endl;
    std::cout << "  num_tokens: " << NUM_TOKENS << std::endl;
    std::cout << "  hidden_size: " << HIDDEN_SIZE << std::endl;
    std::cout << "  num_experts: " << NUM_EXPERTS << std::endl;
    std::cout << "  experts_per_token: " << EXPERTS_PER_TOKEN << std::endl;

    // Allocate host memory
    std::vector<half> h_expanded_output(EXPANDED_TOKENS * HIDDEN_SIZE);
    std::vector<half> h_final_output(NUM_TOKENS * HIDDEN_SIZE);
    std::vector<float> h_scales(NUM_TOKENS * EXPERTS_PER_TOKEN);
    std::vector<int> h_unpermuted_to_permuted(EXPANDED_TOKENS);
    std::vector<int> h_selected_experts(NUM_TOKENS * EXPERTS_PER_TOKEN);

    // Initialize with random values
    srand(42);
    for (int i = 0; i < EXPANDED_TOKENS * HIDDEN_SIZE; i++) {
        h_expanded_output[i] = __float2half((float)(rand() % 200 - 100) / 100.0f);
    }

    // Initialize scales (uniform for testing)
    for (int i = 0; i < NUM_TOKENS * EXPERTS_PER_TOKEN; i++) {
        h_scales[i] = 1.0f / EXPERTS_PER_TOKEN;
    }

    // Initialize permutation (identity for testing)
    for (int i = 0; i < EXPANDED_TOKENS; i++) {
        h_unpermuted_to_permuted[i] = i;
    }

    // Initialize expert selections (sequential for testing)
    for (int t = 0; t < NUM_TOKENS; t++) {
        for (int k = 0; k < EXPERTS_PER_TOKEN; k++) {
            h_selected_experts[t * EXPERTS_PER_TOKEN + k] = k % NUM_EXPERTS;
        }
    }

    // Allocate device memory
    half *d_expanded_output, *d_final_output;
    float *d_scales;
    int *d_unpermuted_to_permuted, *d_selected_experts;

    CUDA_CHECK(cudaMalloc(&d_expanded_output, EXPANDED_TOKENS * HIDDEN_SIZE * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_final_output, NUM_TOKENS * HIDDEN_SIZE * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_scales, NUM_TOKENS * EXPERTS_PER_TOKEN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_unpermuted_to_permuted, EXPANDED_TOKENS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_selected_experts, NUM_TOKENS * EXPERTS_PER_TOKEN * sizeof(int)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_expanded_output, h_expanded_output.data(),
        EXPANDED_TOKENS * HIDDEN_SIZE * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scales, h_scales.data(),
        NUM_TOKENS * EXPERTS_PER_TOKEN * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_unpermuted_to_permuted, h_unpermuted_to_permuted.data(),
        EXPANDED_TOKENS * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_selected_experts, h_selected_experts.data(),
        NUM_TOKENS * EXPERTS_PER_TOKEN * sizeof(int), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Warm-up
    for (int i = 0; i < 3; i++) {
        launchFinalizeMoeRoutingKernel<half>(
            d_expanded_output, d_final_output, nullptr, d_scales,
            d_unpermuted_to_permuted, d_selected_experts,
            NUM_TOKENS, HIDDEN_SIZE, EXPERTS_PER_TOKEN, NUM_EXPERTS, true, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Benchmark
    constexpr int ITERS = 1000;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; i++) {
        launchFinalizeMoeRoutingKernel<half>(
            d_expanded_output, d_final_output, nullptr, d_scales,
            d_unpermuted_to_permuted, d_selected_experts,
            NUM_TOKENS, HIDDEN_SIZE, EXPERTS_PER_TOKEN, NUM_EXPERTS, true, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();

    double avg_us = std::chrono::duration<double, std::micro>(end - start).count() / ITERS;
    double bandwidth_gbps = (double)(EXPANDED_TOKENS * HIDDEN_SIZE * sizeof(half) +
                                     NUM_TOKENS * HIDDEN_SIZE * sizeof(half)) / avg_us / 1000.0;

    std::cout << "\nPerformance:" << std::endl;
    std::cout << "  Average time: " << avg_us << " us" << std::endl;
    std::cout << "  Effective bandwidth: " << bandwidth_gbps << " GB/s" << std::endl;

    // Verify results
    CUDA_CHECK(cudaMemcpy(h_final_output.data(), d_final_output,
        NUM_TOKENS * HIDDEN_SIZE * sizeof(half), cudaMemcpyDeviceToHost));

    // Simple verification: check output is not all zeros
    float sum = 0.0f;
    for (int i = 0; i < 100; i++) {
        sum += fabsf(__half2float(h_final_output[i]));
    }
    bool correct = sum > 0.0f;

    std::cout << "\nVerification: " << (correct ? "PASSED" : "FAILED") << std::endl;
    std::cout << "  Sample output sum: " << sum << std::endl;

    // Cleanup
    CUDA_CHECK(cudaFree(d_expanded_output));
    CUDA_CHECK(cudaFree(d_final_output));
    CUDA_CHECK(cudaFree(d_scales));
    CUDA_CHECK(cudaFree(d_unpermuted_to_permuted));
    CUDA_CHECK(cudaFree(d_selected_experts));
    CUDA_CHECK(cudaStreamDestroy(stream));

    std::cout << "\nTest completed!" << std::endl;
    return 0;
}
