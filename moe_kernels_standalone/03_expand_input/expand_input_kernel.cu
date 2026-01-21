/*
 * MoE Expand Input Rows Kernel - Standalone Version
 *
 * Extracted from TensorRT-LLM moe_kernels.cu
 * This kernel duplicates and permutes input rows based on the expert routing.
 * It expands num_tokens rows to num_tokens * experts_per_token rows.
 *
 * Original function: expandInputRowsKernel
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
// EXPAND_THREADS_PER_BLOCK: Number of threads per block (default: 256)
//   - Each thread handles 128 bits (8 FP16 elements) per iteration
//   - Block size affects occupancy
//
// ELEM_PER_THREAD: Elements processed per thread per iteration
//   - For FP16: 128 bits / 16 bits = 8 elements
//   - For FP32: 128 bits / 32 bits = 4 elements
//   - For FP4/FP8: Different values based on quantization
// ============================================================================

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

constexpr int EXPAND_THREADS_PER_BLOCK = 256;

// Helper: find which expert a token belongs to based on offset array
__device__ __forceinline__ int findExpertForToken(
    int64_t const* expert_first_token_offset,
    int num_experts,
    int64_t token)
{
    // Binary search to find expert
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
// Expand Input Rows Kernel (FP16 version)
// ============================================================================
template <typename T>
__global__ void expandInputRowsKernel(
    T const* unpermuted_input,                    // [num_tokens, hidden_size]
    T* permuted_output,                           // [num_tokens * k, hidden_size]
    float const* unpermuted_scales,               // [num_tokens, k] or nullptr
    float* permuted_scales,                       // [num_tokens * k] or nullptr
    int const* permuted_row_to_unpermuted_row,    // [num_tokens * k]
    int64_t const num_tokens,
    int64_t const hidden_size,
    int64_t const k,                              // experts_per_token
    int64_t const* expert_first_token_offset,     // [num_experts + 1]
    int64_t const num_experts_per_node)
{
    // Calculate number of valid tokens (tokens assigned to local experts)
    int64_t const num_valid_tokens = expert_first_token_offset[num_experts_per_node];

    // Each block handles one permuted row
    for (int64_t permuted_row = blockIdx.x; permuted_row < num_valid_tokens; permuted_row += gridDim.x)
    {
        // Get the unpermuted row index
        int64_t const unpermuted_row = permuted_row_to_unpermuted_row[permuted_row];

        // Load 128-bits per thread (8 FP16 elements)
        constexpr int64_t ELEM_PER_THREAD = 128 / (sizeof(T) * 8);  // bits / bits_per_elem

        using DataElem = typename std::conditional<sizeof(T) == 2,
            float4,  // 4 floats = 16 bytes = 8 FP16
            float4   // same for FP32 (4 elements)
        >::type;

        // Duplicate and permute rows
        int64_t const source_k_rank = unpermuted_row / num_tokens;
        int64_t const source_row = unpermuted_row % num_tokens;

        auto const* source_row_ptr = reinterpret_cast<DataElem const*>(
            unpermuted_input + source_row * hidden_size);
        auto* dest_row_ptr = reinterpret_cast<DataElem*>(
            permuted_output + permuted_row * hidden_size);

        int64_t const start_offset = threadIdx.x;
        int64_t const stride = EXPAND_THREADS_PER_BLOCK;
        int64_t const num_elems_in_col = hidden_size / ELEM_PER_THREAD;
        assert(hidden_size % ELEM_PER_THREAD == 0);

        // Copy input row to permuted position
        for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride)
        {
            dest_row_ptr[elem_index] = source_row_ptr[elem_index];
        }

        // Copy scales if provided
        if (permuted_scales && threadIdx.x == 0)
        {
            int64_t const source_k_idx = source_row * k + source_k_rank;
            permuted_scales[permuted_row] = unpermuted_scales ? unpermuted_scales[source_k_idx] : 1.0f;
        }
    }
}

// ============================================================================
// Launcher function
// ============================================================================
template <typename T>
void launchExpandInputRowsKernel(
    T const* unpermuted_input,
    T* permuted_output,
    float const* unpermuted_scales,
    float* permuted_scales,
    int const* permuted_row_to_unpermuted_row,
    int64_t num_tokens,
    int64_t hidden_size,
    int experts_per_token,
    int num_experts,
    int64_t* expert_first_token_offset,
    cudaStream_t stream)
{
    // Get number of SMs for optimal grid size
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    int64_t num_expanded_tokens = num_tokens * experts_per_token;
    int blocks = std::min((int)num_expanded_tokens, props.multiProcessorCount * 4);
    int threads = EXPAND_THREADS_PER_BLOCK;

    expandInputRowsKernel<T><<<blocks, threads, 0, stream>>>(
        unpermuted_input, permuted_output, unpermuted_scales, permuted_scales,
        permuted_row_to_unpermuted_row, num_tokens, hidden_size, experts_per_token,
        expert_first_token_offset, num_experts);
}

// Explicit instantiations
template void launchExpandInputRowsKernel<half>(
    half const*, half*, float const*, float*, int const*,
    int64_t, int64_t, int, int, int64_t*, cudaStream_t);

template void launchExpandInputRowsKernel<float>(
    float const*, float*, float const*, float*, int const*,
    int64_t, int64_t, int, int, int64_t*, cudaStream_t);

// ============================================================================
// Test program
// ============================================================================
int main() {
    std::cout << "=== MoE Expand Input Rows Kernel Test ===" << std::endl;

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
    std::vector<half> h_input(NUM_TOKENS * HIDDEN_SIZE);
    std::vector<half> h_output(EXPANDED_TOKENS * HIDDEN_SIZE);
    std::vector<float> h_input_scales(NUM_TOKENS * EXPERTS_PER_TOKEN);
    std::vector<float> h_output_scales(EXPANDED_TOKENS);
    std::vector<int> h_permuted_to_unpermuted(EXPANDED_TOKENS);
    std::vector<int64_t> h_expert_offset(NUM_EXPERTS + 1);

    // Initialize input with some values
    srand(42);
    for (int i = 0; i < NUM_TOKENS * HIDDEN_SIZE; i++) {
        h_input[i] = __float2half((float)(rand() % 100) / 100.0f);
    }
    for (int i = 0; i < NUM_TOKENS * EXPERTS_PER_TOKEN; i++) {
        h_input_scales[i] = 1.0f / EXPERTS_PER_TOKEN;
    }

    // Create simple permutation (identity for testing)
    for (int i = 0; i < EXPANDED_TOKENS; i++) {
        h_permuted_to_unpermuted[i] = i;
    }

    // Create uniform expert distribution
    int tokens_per_expert = EXPANDED_TOKENS / NUM_EXPERTS;
    for (int i = 0; i <= NUM_EXPERTS; i++) {
        h_expert_offset[i] = i * tokens_per_expert;
    }

    // Allocate device memory
    half *d_input, *d_output;
    float *d_input_scales, *d_output_scales;
    int *d_permuted_to_unpermuted;
    int64_t *d_expert_offset;

    CUDA_CHECK(cudaMalloc(&d_input, NUM_TOKENS * HIDDEN_SIZE * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, EXPANDED_TOKENS * HIDDEN_SIZE * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_input_scales, NUM_TOKENS * EXPERTS_PER_TOKEN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output_scales, EXPANDED_TOKENS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_permuted_to_unpermuted, EXPANDED_TOKENS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_expert_offset, (NUM_EXPERTS + 1) * sizeof(int64_t)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), NUM_TOKENS * HIDDEN_SIZE * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input_scales, h_input_scales.data(), NUM_TOKENS * EXPERTS_PER_TOKEN * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_permuted_to_unpermuted, h_permuted_to_unpermuted.data(), EXPANDED_TOKENS * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_expert_offset, h_expert_offset.data(), (NUM_EXPERTS + 1) * sizeof(int64_t), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Warm-up
    for (int i = 0; i < 3; i++) {
        launchExpandInputRowsKernel<half>(
            d_input, d_output, d_input_scales, d_output_scales,
            d_permuted_to_unpermuted, NUM_TOKENS, HIDDEN_SIZE,
            EXPERTS_PER_TOKEN, NUM_EXPERTS, d_expert_offset, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Benchmark
    constexpr int ITERS = 1000;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; i++) {
        launchExpandInputRowsKernel<half>(
            d_input, d_output, d_input_scales, d_output_scales,
            d_permuted_to_unpermuted, NUM_TOKENS, HIDDEN_SIZE,
            EXPERTS_PER_TOKEN, NUM_EXPERTS, d_expert_offset, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();

    double avg_us = std::chrono::duration<double, std::micro>(end - start).count() / ITERS;
    double bandwidth_gbps = (double)(NUM_TOKENS * HIDDEN_SIZE * sizeof(half) +
                                     EXPANDED_TOKENS * HIDDEN_SIZE * sizeof(half)) / avg_us / 1000.0;

    std::cout << "\nPerformance:" << std::endl;
    std::cout << "  Average time: " << avg_us << " us" << std::endl;
    std::cout << "  Effective bandwidth: " << bandwidth_gbps << " GB/s" << std::endl;

    // Verify results
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, EXPANDED_TOKENS * HIDDEN_SIZE * sizeof(half), cudaMemcpyDeviceToHost));

    bool correct = true;
    for (int row = 0; row < std::min(10, EXPANDED_TOKENS); row++) {
        int unpermuted_row = h_permuted_to_unpermuted[row];
        int source_row = unpermuted_row % NUM_TOKENS;
        for (int col = 0; col < 4; col++) {  // Check first 4 elements
            half expected = h_input[source_row * HIDDEN_SIZE + col];
            half actual = h_output[row * HIDDEN_SIZE + col];
            if (__half2float(expected) != __half2float(actual)) {
                std::cout << "Mismatch at row " << row << " col " << col << std::endl;
                correct = false;
            }
        }
    }

    std::cout << "\nVerification: " << (correct ? "PASSED" : "FAILED") << std::endl;

    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_input_scales));
    CUDA_CHECK(cudaFree(d_output_scales));
    CUDA_CHECK(cudaFree(d_permuted_to_unpermuted));
    CUDA_CHECK(cudaFree(d_expert_offset));
    CUDA_CHECK(cudaStreamDestroy(stream));

    std::cout << "\nTest completed!" << std::endl;
    return 0;
}
