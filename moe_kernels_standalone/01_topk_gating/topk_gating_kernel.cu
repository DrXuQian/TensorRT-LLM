/*
 * MoE TopK Gating Kernel - Standalone Version
 *
 * Extracted from TensorRT-LLM customMoeRoutingKernels.cu
 * This kernel computes TopK expert selection and softmax routing weights.
 *
 * Original function: customMoeRoutingKernel
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cassert>
#include <cmath>

namespace cg = cooperative_groups;

// ============================================================================
// Tunable Parameters
// ============================================================================
// BLOCK_SIZE: Number of threads per block (default: 1024)
//   - Must be multiple of WARP_SIZE (32)
//   - Each warp handles one token
//
// WARP_SIZE: 32 (fixed by hardware)
//
// MaxNumExperts: Template parameter (32, 64, 96, 128)
//   - Determines register allocation for scores
//   - Must be >= actual num_experts
//
// MaxNumTopExperts: Template parameter (1, 2, 4, 8)
//   - Number of experts to select per token
//   - Determines TopK storage
//
// DoSoftmaxBeforeTopK: Whether to apply softmax before TopK selection
//   - true: softmax(scores), then select top-k
//   - false: select top-k scores, then softmax on selected
// ============================================================================

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

constexpr int BLOCK_SIZE = 1024;
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;

// ============================================================================
// TopK Reduction Utilities
// ============================================================================

// Pack value and index into single 64-bit value for efficient max reduction
struct TopKPackedType {
    uint64_t packed;

    __device__ TopKPackedType() : packed(0) {}

    __device__ TopKPackedType(float val, int idx) {
        // Convert float to sortable uint32
        uint32_t valBits = __float_as_uint(val);
        // Flip sign bit for proper ordering
        valBits = (valBits & 0x80000000) ? ~valBits : (valBits | 0x80000000);
        // Pack: high 32 bits = value, low 32 bits = negated index (for stability)
        packed = (uint64_t(valBits) << 32) | uint32_t(65535 - idx);
    }

    __device__ void unpack(float& val, int& idx) const {
        idx = 65535 - int(packed & 0xFFFF);
        uint32_t valBits = uint32_t(packed >> 32);
        valBits = (valBits & 0x80000000) ? (valBits & 0x7FFFFFFF) : ~valBits;
        val = __uint_as_float(valBits);
    }

    __device__ uint64_t warpMax(cg::thread_block_tile<WARP_SIZE> const& warp) {
        return cg::reduce(warp, packed, cg::greater<uint64_t>{});
    }
};

// ============================================================================
// Softmax computation within warp
// ============================================================================
template <int VecSize>
__device__ void warpSoftmax(cg::thread_block_tile<WARP_SIZE> const& warp, float (&scores)[VecSize]) {
    // Find max across warp
    float maxScore = -INFINITY;
    #pragma unroll
    for (int i = 0; i < VecSize; ++i) {
        maxScore = fmaxf(maxScore, scores[i]);
    }
    maxScore = cg::reduce(warp, maxScore, cg::greater<float>{});

    // Compute exp and sum
    float sumScore = 0.0f;
    #pragma unroll
    for (int i = 0; i < VecSize; ++i) {
        scores[i] = expf(scores[i] - maxScore);
        sumScore += scores[i];
    }
    sumScore = cg::reduce(warp, sumScore, cg::plus<float>{});

    // Normalize
    #pragma unroll
    for (int i = 0; i < VecSize; ++i) {
        scores[i] /= sumScore;
    }
}

__device__ float warpSoftmaxSingle(cg::thread_block_tile<WARP_SIZE> const& warp,
                                    float score, int laneIdx, int numTopExperts) {
    float maxScore = (laneIdx < numTopExperts) ? score : -INFINITY;
    maxScore = cg::reduce(warp, maxScore, cg::greater<float>{});

    float expScore = (laneIdx < numTopExperts) ? expf(score - maxScore) : 0.0f;
    float sumScore = cg::reduce(warp, expScore, cg::plus<float>{});

    return expScore / sumScore;
}

// ============================================================================
// TopK Gating Kernel
// ============================================================================
template <int MaxNumExperts, int MaxNumTopExperts, bool DoSoftmaxBeforeTopK>
__global__ void topkGatingKernel(
    float const* routerLogits,    // [num_tokens, num_experts]
    float* topkValues,            // [num_tokens, topK]
    int* topkIndices,             // [num_tokens, topK]
    int numTokens,
    int numExperts,
    int topK)
{
    int const blockRank = blockIdx.x;
    int const tIdx = BLOCK_SIZE * blockRank + threadIdx.x;
    int const warpIdx = tIdx / WARP_SIZE;
    int const laneIdx = tIdx % WARP_SIZE;
    int const warpNum = gridDim.x * WARPS_PER_BLOCK;

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WARP_SIZE>(block);

    constexpr int SCORES_PER_THREAD = MaxNumExperts / WARP_SIZE;

    // Process tokens - each warp handles one token
    for (int tokenId = warpIdx; tokenId < numTokens; tokenId += warpNum) {
        int scoreOffset = tokenId * numExperts;
        int outputOffset = tokenId * topK;

        // Load scores for this token
        float inputScore[SCORES_PER_THREAD];
        int inputIndex[SCORES_PER_THREAD];

        #pragma unroll
        for (int i = 0; i < SCORES_PER_THREAD; ++i) {
            int expertIdx = i * WARP_SIZE + laneIdx;
            if (expertIdx < numExperts) {
                inputScore[i] = routerLogits[scoreOffset + expertIdx];
            } else {
                inputScore[i] = -INFINITY;
            }
            inputIndex[i] = expertIdx;
        }

        // Apply softmax before TopK if requested
        if constexpr (DoSoftmaxBeforeTopK) {
            warpSoftmax(warp, inputScore);
        }

        // Find TopK using warp reduction
        float topKScore[MaxNumTopExperts];
        int topKIdx[MaxNumTopExperts];

        // Initialize
        #pragma unroll
        for (int k = 0; k < MaxNumTopExperts; ++k) {
            topKScore[k] = -INFINITY;
            topKIdx[k] = -1;
        }

        // Simple TopK: find max K times
        for (int k = 0; k < topK; ++k) {
            // Find current max across all threads
            TopKPackedType bestLocal;
            float bestScore = -INFINITY;
            int bestIdx = -1;

            #pragma unroll
            for (int i = 0; i < SCORES_PER_THREAD; ++i) {
                if (inputScore[i] > bestScore) {
                    bestScore = inputScore[i];
                    bestIdx = inputIndex[i];
                }
            }

            TopKPackedType packed(bestScore, bestIdx);
            uint64_t maxPacked = packed.warpMax(warp);

            // Unpack winner
            TopKPackedType winner;
            winner.packed = maxPacked;
            winner.unpack(topKScore[k], topKIdx[k]);

            // Invalidate winner for next iteration
            #pragma unroll
            for (int i = 0; i < SCORES_PER_THREAD; ++i) {
                if (inputIndex[i] == topKIdx[k]) {
                    inputScore[i] = -INFINITY;
                }
            }
        }

        // Apply softmax after TopK if requested
        if constexpr (!DoSoftmaxBeforeTopK) {
            if (laneIdx < topK) {
                topKScore[laneIdx] = warpSoftmaxSingle(warp, topKScore[laneIdx], laneIdx, topK);
            }
        }

        // Write output (first topK threads)
        if (laneIdx < topK) {
            topkValues[outputOffset + laneIdx] = topKScore[laneIdx];
            topkIndices[outputOffset + laneIdx] = topKIdx[laneIdx];
        }
    }
}

// ============================================================================
// Launcher function
// ============================================================================
void launchTopKGatingKernel(
    float const* routerLogits,
    float* topkValues,
    int* topkIndices,
    int numTokens,
    int numExperts,
    int topK,
    bool doSoftmaxBeforeTopK,
    cudaStream_t stream)
{
    int maxBlocks = 1024;
    int numBlocks = std::min((numTokens - 1) / WARPS_PER_BLOCK + 1, maxBlocks);

    // Select kernel based on parameters
    // For simplicity, we instantiate common cases
    if (numExperts <= 128 && topK <= 8) {
        if (doSoftmaxBeforeTopK) {
            topkGatingKernel<128, 8, true><<<numBlocks, BLOCK_SIZE, 0, stream>>>(
                routerLogits, topkValues, topkIndices, numTokens, numExperts, topK);
        } else {
            topkGatingKernel<128, 8, false><<<numBlocks, BLOCK_SIZE, 0, stream>>>(
                routerLogits, topkValues, topkIndices, numTokens, numExperts, topK);
        }
    } else {
        std::cerr << "Unsupported configuration: numExperts=" << numExperts << ", topK=" << topK << std::endl;
    }
}

// ============================================================================
// Test program
// ============================================================================
int main() {
    std::cout << "=== MoE TopK Gating Kernel Test ===" << std::endl;

    // Test configuration
    constexpr int NUM_TOKENS = 3823;  // User's test case
    constexpr int NUM_EXPERTS = 128;
    constexpr int TOP_K = 8;

    std::cout << "Configuration:" << std::endl;
    std::cout << "  num_tokens: " << NUM_TOKENS << std::endl;
    std::cout << "  num_experts: " << NUM_EXPERTS << std::endl;
    std::cout << "  top_k: " << TOP_K << std::endl;

    // Allocate host memory
    std::vector<float> h_logits(NUM_TOKENS * NUM_EXPERTS);
    std::vector<float> h_topk_values(NUM_TOKENS * TOP_K);
    std::vector<int> h_topk_indices(NUM_TOKENS * TOP_K);

    // Initialize logits with random values
    srand(42);
    for (int i = 0; i < NUM_TOKENS * NUM_EXPERTS; i++) {
        h_logits[i] = (float)(rand() % 1000) / 100.0f - 5.0f;  // Range: -5 to 5
    }

    // Allocate device memory
    float *d_logits, *d_topk_values;
    int *d_topk_indices;

    CUDA_CHECK(cudaMalloc(&d_logits, NUM_TOKENS * NUM_EXPERTS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_topk_values, NUM_TOKENS * TOP_K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_topk_indices, NUM_TOKENS * TOP_K * sizeof(int)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_logits, h_logits.data(),
        NUM_TOKENS * NUM_EXPERTS * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Warm-up
    for (int i = 0; i < 3; i++) {
        launchTopKGatingKernel(d_logits, d_topk_values, d_topk_indices,
            NUM_TOKENS, NUM_EXPERTS, TOP_K, false, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Benchmark
    constexpr int ITERS = 1000;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; i++) {
        launchTopKGatingKernel(d_logits, d_topk_values, d_topk_indices,
            NUM_TOKENS, NUM_EXPERTS, TOP_K, false, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();

    double avg_us = std::chrono::duration<double, std::micro>(end - start).count() / ITERS;

    std::cout << "\nPerformance:" << std::endl;
    std::cout << "  Average time: " << avg_us << " us" << std::endl;

    // Copy results back
    CUDA_CHECK(cudaMemcpy(h_topk_values.data(), d_topk_values,
        NUM_TOKENS * TOP_K * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_topk_indices.data(), d_topk_indices,
        NUM_TOKENS * TOP_K * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify results
    std::cout << "\nSample outputs (first 3 tokens):" << std::endl;
    for (int t = 0; t < 3; t++) {
        std::cout << "  Token " << t << ": experts=[";
        float sum = 0.0f;
        for (int k = 0; k < TOP_K; k++) {
            std::cout << h_topk_indices[t * TOP_K + k];
            if (k < TOP_K - 1) std::cout << ",";
            sum += h_topk_values[t * TOP_K + k];
        }
        std::cout << "], weight_sum=" << sum << std::endl;
    }

    // Check softmax normalization (sum should be ~1.0)
    bool correct = true;
    for (int t = 0; t < std::min(100, NUM_TOKENS); t++) {
        float sum = 0.0f;
        for (int k = 0; k < TOP_K; k++) {
            sum += h_topk_values[t * TOP_K + k];
        }
        if (fabsf(sum - 1.0f) > 0.01f) {
            std::cout << "Warning: Token " << t << " weight sum = " << sum << " (expected ~1.0)" << std::endl;
            correct = false;
        }
    }

    std::cout << "\nVerification: " << (correct ? "PASSED" : "FAILED") << std::endl;

    // Cleanup
    CUDA_CHECK(cudaFree(d_logits));
    CUDA_CHECK(cudaFree(d_topk_values));
    CUDA_CHECK(cudaFree(d_topk_indices));
    CUDA_CHECK(cudaStreamDestroy(stream));

    std::cout << "\nTest completed!" << std::endl;
    return 0;
}
