/*
 * MoE Permute/Sort Kernel - Standalone Version
 *
 * Extracted from TensorRT-LLM moe_kernels.cu
 * This kernel sorts tokens by their selected expert IDs and builds
 * the permutation maps for the MoE computation.
 *
 * Original functions:
 * - fusedBuildExpertMapsSortFirstTokenKernel
 * - threeStepBuildExpertMapsSortFirstToken (fallback for large token counts)
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cassert>

// ============================================================================
// Tunable Parameters
// ============================================================================
// BLOCK_SIZE: Number of threads per block (32, 64, 128, 256)
//   - Larger block size can handle more tokens in fused kernel
//   - Must be power of 2
//   - Fused kernel limited to num_tokens <= 256
//
// LOG2_NUM_EXPERTS: log2(num_experts + 2), determines radix sort bits
//   - For 128 experts: log2(130) = 8 bits
//   - For 256 experts: log2(258) = 9 bits
//   - Maximum supported: 9 (up to 510 experts)
//
// EXPERTS_PER_TOKEN: Number of experts selected per token (k)
//   - Supported values: 1, 2, 4, 6, 8
// ============================================================================

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

// ============================================================================
// Fused Permute/Sort Kernel (for small token counts <= 256)
// Uses CUB BlockRadixRank for efficient in-block sorting
// ============================================================================
template <int BLOCK_SIZE, int EXPERTS_PER_TOKEN, int LOG2_NUM_EXPERTS>
__global__ void fusedBuildExpertMapsSortFirstTokenKernel(
    int const* const token_selected_experts,      // [num_tokens, experts_per_token]
    int* const permuted_row_to_unpermuted_row,    // [num_tokens * experts_per_token]
    int* const unpermuted_row_to_permuted_row,    // [num_tokens * experts_per_token]
    int64_t* const expert_first_token_offset,     // [num_experts + 1]
    int64_t const num_tokens,
    int const experts_per_token,
    int const start_expert,
    int const end_expert,
    int const num_experts_per_node)
{
    // Only using block wise collective so we can only have one block
    assert(gridDim.x == 1);
    assert(start_expert <= end_expert);
    assert(num_experts_per_node == (end_expert - start_expert));
    assert(num_experts_per_node <= (1 << LOG2_NUM_EXPERTS));

    int const token = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    bool is_valid_token = token < num_tokens;

    // Local storage for this token's expert selections
    int local_token_selected_experts[EXPERTS_PER_TOKEN];
    int local_token_permuted_indices[EXPERTS_PER_TOKEN];

    // Build expert map for this token
    #pragma unroll
    for (int i = 0; i < EXPERTS_PER_TOKEN; i++) {
        int const expert = is_valid_token ?
            token_selected_experts[token * EXPERTS_PER_TOKEN + i] : num_experts_per_node;

        bool is_valid_expert = expert >= start_expert && expert < end_expert;
        local_token_selected_experts[i] = !is_valid_token ? num_experts_per_node + 1
            : is_valid_expert ? (expert - start_expert)
            : num_experts_per_node;
    }

    // Use CUB BlockRadixRank for sorting
    using BlockRadixRank = cub::BlockRadixRank<BLOCK_SIZE, LOG2_NUM_EXPERTS, false>;
    extern __shared__ unsigned char temp_storage[];
    auto& sort_temp = *reinterpret_cast<typename BlockRadixRank::TempStorage*>(temp_storage);

    static_assert(BlockRadixRank::BINS_TRACKED_PER_THREAD * BLOCK_SIZE >= (1 << LOG2_NUM_EXPERTS));

    int local_expert_first_token_offset[BlockRadixRank::BINS_TRACKED_PER_THREAD];

    cub::BFEDigitExtractor<int> extractor(0, LOG2_NUM_EXPERTS);
    BlockRadixRank(sort_temp).RankKeys(
        local_token_selected_experts, local_token_permuted_indices,
        extractor, local_expert_first_token_offset);

    // Write permutation maps
    if (is_valid_token) {
        #pragma unroll
        for (int i = 0; i < EXPERTS_PER_TOKEN; i++) {
            int const unpermuted_row = i * num_tokens + token;
            int const permuted_row = local_token_permuted_indices[i];
            permuted_row_to_unpermuted_row[permuted_row] = unpermuted_row;
            unpermuted_row_to_permuted_row[unpermuted_row] = permuted_row;
        }
    }

    // Write expert offsets
    #pragma unroll
    for (int expert_id = 0; expert_id < BlockRadixRank::BINS_TRACKED_PER_THREAD; expert_id++) {
        int out_expert_id = expert_id + token * BlockRadixRank::BINS_TRACKED_PER_THREAD;
        if (out_expert_id < num_experts_per_node + 1) {
            expert_first_token_offset[out_expert_id] = local_expert_first_token_offset[expert_id];
        }
    }
}

// ============================================================================
// Three-step Sort Kernels (for large token counts > 256)
// Step 1: Block-level prefix sum
// Step 2: Global prefix sum
// Step 3: Merge and write final permutation
// ============================================================================

// Step 1: Count tokens per expert in each block
template <int BLOCK_SIZE>
__global__ void blockExpertPrefixSumKernel(
    int const* token_selected_experts,
    int* blocked_expert_counts,
    int64_t const num_tokens,
    int const num_experts_per_node,
    int const experts_per_token,
    int const start_expert,
    int const end_expert)
{
    int const block_id = blockIdx.x;
    int const tid = threadIdx.x;
    int const token_start = block_id * BLOCK_SIZE;

    // Count experts in this block
    extern __shared__ int shared_counts[];

    // Initialize shared memory counts
    for (int e = tid; e < num_experts_per_node + 1; e += BLOCK_SIZE) {
        shared_counts[e] = 0;
    }
    __syncthreads();

    // Count tokens for each expert
    int const token = token_start + tid;
    if (token < num_tokens) {
        for (int k = 0; k < experts_per_token; k++) {
            int expert = token_selected_experts[token * experts_per_token + k];
            if (expert >= start_expert && expert < end_expert) {
                atomicAdd(&shared_counts[expert - start_expert], 1);
            } else {
                atomicAdd(&shared_counts[num_experts_per_node], 1);
            }
        }
    }
    __syncthreads();

    // Write to global memory
    int const num_blocks = (num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int e = tid; e < num_experts_per_node + 1; e += BLOCK_SIZE) {
        blocked_expert_counts[e * num_blocks + block_id] = shared_counts[e];
    }
}

// Step 2: Global prefix sum across blocks
__global__ void globalExpertPrefixSumKernel(
    int const* blocked_expert_counts,
    int* blocked_expert_counts_cumsum,
    int64_t* expert_first_token_offset,
    int const num_blocks,
    int const num_experts_per_node)
{
    int const expert = blockIdx.x;
    if (expert > num_experts_per_node) return;

    int sum = 0;
    for (int b = 0; b < num_blocks; b++) {
        blocked_expert_counts_cumsum[expert * num_blocks + b] = sum;
        sum += blocked_expert_counts[expert * num_blocks + b];
    }

    if (threadIdx.x == 0) {
        expert_first_token_offset[expert] = sum;
    }
}

// Step 3: Merge results and write final permutation
template <int BLOCK_SIZE>
__global__ void mergeExpertPrefixSumKernel(
    int const* token_selected_experts,
    int const* blocked_expert_counts,
    int const* blocked_expert_counts_cumsum,
    int64_t const* expert_first_token_offset,
    int* permuted_row_to_unpermuted_row,
    int* unpermuted_row_to_permuted_row,
    int64_t const num_tokens,
    int const num_experts_per_node,
    int const experts_per_token,
    int const start_expert,
    int const end_expert)
{
    int const block_id = blockIdx.x;
    int const tid = threadIdx.x;
    int const token_start = block_id * BLOCK_SIZE;
    int const num_blocks = (num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE;

    extern __shared__ int shared_data[];
    int* local_counts = shared_data;
    int* local_offsets = shared_data + num_experts_per_node + 1;

    // Load block-local counts and compute prefix sum
    for (int e = tid; e < num_experts_per_node + 1; e += BLOCK_SIZE) {
        int global_offset = 0;
        for (int prev_e = 0; prev_e < e; prev_e++) {
            global_offset += expert_first_token_offset[prev_e];
        }
        local_offsets[e] = global_offset + blocked_expert_counts_cumsum[e * num_blocks + block_id];
        local_counts[e] = 0;
    }
    __syncthreads();

    // Assign permuted indices
    int const token = token_start + tid;
    if (token < num_tokens) {
        for (int k = 0; k < experts_per_token; k++) {
            int expert = token_selected_experts[token * experts_per_token + k];
            int local_expert;
            if (expert >= start_expert && expert < end_expert) {
                local_expert = expert - start_expert;
            } else {
                local_expert = num_experts_per_node;
            }

            int local_idx = atomicAdd(&local_counts[local_expert], 1);
            int permuted_row = local_offsets[local_expert] + local_idx;
            int unpermuted_row = k * num_tokens + token;

            permuted_row_to_unpermuted_row[permuted_row] = unpermuted_row;
            unpermuted_row_to_permuted_row[unpermuted_row] = permuted_row;
        }
    }
}

// ============================================================================
// Launcher function
// ============================================================================
void launchPermuteSortKernel(
    int const* token_selected_experts,
    int* permuted_row_to_unpermuted_row,
    int* unpermuted_row_to_permuted_row,
    int64_t* expert_first_token_offset,
    int64_t num_tokens,
    int num_experts,
    int experts_per_token,
    cudaStream_t stream)
{
    int const start_expert = 0;
    int const end_expert = num_experts;
    int const num_experts_per_node = num_experts;

    // For small token counts, use fused kernel
    if (num_tokens <= 256) {
        // Determine block size
        int block_size = 32;
        if (num_tokens > 32) block_size = 64;
        if (num_tokens > 64) block_size = 128;
        if (num_tokens > 128) block_size = 256;

        // Calculate LOG2_NUM_EXPERTS
        int log2_experts = static_cast<int>(log2(num_experts + 1)) + 1;

        // Use template specialization based on parameters
        // For simplicity, we show one case here
        if (block_size == 256 && experts_per_token == 8 && log2_experts == 8) {
            using BlockRadixRank = cub::BlockRadixRank<256, 8, false>;
            size_t shared_size = sizeof(typename BlockRadixRank::TempStorage);

            fusedBuildExpertMapsSortFirstTokenKernel<256, 8, 8><<<1, 256, shared_size, stream>>>(
                token_selected_experts, permuted_row_to_unpermuted_row,
                unpermuted_row_to_permuted_row, expert_first_token_offset,
                num_tokens, experts_per_token, start_expert, end_expert, num_experts_per_node);
        }
        // Add more template instantiations as needed...
    } else {
        // Use three-step approach for large token counts
        constexpr int BLOCK_SIZE = 256;
        int num_blocks = (num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Allocate temporary buffers
        int *blocked_expert_counts, *blocked_expert_counts_cumsum;
        CUDA_CHECK(cudaMalloc(&blocked_expert_counts,
            (num_experts_per_node + 1) * num_blocks * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&blocked_expert_counts_cumsum,
            (num_experts_per_node + 1) * num_blocks * sizeof(int)));

        // Step 1: Block-level counting
        size_t shared_size1 = (num_experts_per_node + 1) * sizeof(int);
        blockExpertPrefixSumKernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE, shared_size1, stream>>>(
            token_selected_experts, blocked_expert_counts, num_tokens,
            num_experts_per_node, experts_per_token, start_expert, end_expert);

        // Step 2: Global prefix sum
        globalExpertPrefixSumKernel<<<num_experts_per_node + 1, 1, 0, stream>>>(
            blocked_expert_counts, blocked_expert_counts_cumsum,
            expert_first_token_offset, num_blocks, num_experts_per_node);

        // Step 3: Merge and write permutation
        size_t shared_size3 = 2 * (num_experts_per_node + 1) * sizeof(int);
        mergeExpertPrefixSumKernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE, shared_size3, stream>>>(
            token_selected_experts, blocked_expert_counts, blocked_expert_counts_cumsum,
            expert_first_token_offset, permuted_row_to_unpermuted_row,
            unpermuted_row_to_permuted_row, num_tokens, num_experts_per_node,
            experts_per_token, start_expert, end_expert);

        // Cleanup
        CUDA_CHECK(cudaFree(blocked_expert_counts));
        CUDA_CHECK(cudaFree(blocked_expert_counts_cumsum));
    }
}

// ============================================================================
// Test program
// ============================================================================
int main() {
    std::cout << "=== MoE Permute/Sort Kernel Test ===" << std::endl;

    // Test configuration
    constexpr int NUM_TOKENS = 128;  // Small enough for fused kernel
    constexpr int NUM_EXPERTS = 128;
    constexpr int EXPERTS_PER_TOKEN = 8;

    std::cout << "Configuration:" << std::endl;
    std::cout << "  num_tokens: " << NUM_TOKENS << std::endl;
    std::cout << "  num_experts: " << NUM_EXPERTS << std::endl;
    std::cout << "  experts_per_token: " << EXPERTS_PER_TOKEN << std::endl;

    // Allocate host memory
    std::vector<int> h_selected_experts(NUM_TOKENS * EXPERTS_PER_TOKEN);
    std::vector<int> h_permuted_to_unpermuted(NUM_TOKENS * EXPERTS_PER_TOKEN);
    std::vector<int> h_unpermuted_to_permuted(NUM_TOKENS * EXPERTS_PER_TOKEN);
    std::vector<int64_t> h_expert_offset(NUM_EXPERTS + 1);

    // Initialize with random expert selections
    srand(42);
    for (int t = 0; t < NUM_TOKENS; t++) {
        for (int k = 0; k < EXPERTS_PER_TOKEN; k++) {
            h_selected_experts[t * EXPERTS_PER_TOKEN + k] = rand() % NUM_EXPERTS;
        }
    }

    // Allocate device memory
    int *d_selected_experts, *d_permuted_to_unpermuted, *d_unpermuted_to_permuted;
    int64_t *d_expert_offset;

    CUDA_CHECK(cudaMalloc(&d_selected_experts, NUM_TOKENS * EXPERTS_PER_TOKEN * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_permuted_to_unpermuted, NUM_TOKENS * EXPERTS_PER_TOKEN * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_unpermuted_to_permuted, NUM_TOKENS * EXPERTS_PER_TOKEN * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_expert_offset, (NUM_EXPERTS + 1) * sizeof(int64_t)));

    CUDA_CHECK(cudaMemcpy(d_selected_experts, h_selected_experts.data(),
        NUM_TOKENS * EXPERTS_PER_TOKEN * sizeof(int), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Warm-up
    for (int i = 0; i < 3; i++) {
        launchPermuteSortKernel(d_selected_experts, d_permuted_to_unpermuted,
            d_unpermuted_to_permuted, d_expert_offset, NUM_TOKENS,
            NUM_EXPERTS, EXPERTS_PER_TOKEN, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Benchmark
    constexpr int ITERS = 1000;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; i++) {
        launchPermuteSortKernel(d_selected_experts, d_permuted_to_unpermuted,
            d_unpermuted_to_permuted, d_expert_offset, NUM_TOKENS,
            NUM_EXPERTS, EXPERTS_PER_TOKEN, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();

    double avg_us = std::chrono::duration<double, std::micro>(end - start).count() / ITERS;
    std::cout << "\nPerformance:" << std::endl;
    std::cout << "  Average time: " << avg_us << " us" << std::endl;

    // Copy results back
    CUDA_CHECK(cudaMemcpy(h_expert_offset.data(), d_expert_offset,
        (NUM_EXPERTS + 1) * sizeof(int64_t), cudaMemcpyDeviceToHost));

    // Verify results
    std::cout << "\nVerification:" << std::endl;
    int total_tokens = 0;
    for (int e = 0; e < NUM_EXPERTS; e++) {
        total_tokens += h_expert_offset[e];
    }
    std::cout << "  Total tokens assigned: " << total_tokens << std::endl;
    std::cout << "  Expected: " << NUM_TOKENS * EXPERTS_PER_TOKEN << std::endl;

    // Cleanup
    CUDA_CHECK(cudaFree(d_selected_experts));
    CUDA_CHECK(cudaFree(d_permuted_to_unpermuted));
    CUDA_CHECK(cudaFree(d_unpermuted_to_permuted));
    CUDA_CHECK(cudaFree(d_expert_offset));
    CUDA_CHECK(cudaStreamDestroy(stream));

    std::cout << "\nTest completed!" << std::endl;
    return 0;
}
