# MoE TopK Gating Kernel

## Overview

This kernel implements the MoE (Mixture of Experts) routing computation. It takes router logits and produces:
1. Top-K expert indices for each token
2. Softmax-normalized routing weights

**Original source:** `cpp/tensorrt_llm/kernels/customMoeRoutingKernels.cu`

## What this kernel does

1. Load router logits for each token (shape: [num_tokens, num_experts])
2. Optionally apply softmax to all logits
3. Find top-K highest scoring experts using warp-level reduction
4. Apply softmax to selected scores (if not done before)
5. Output expert indices and routing weights

## Build

```bash
mkdir build && cd build
cmake ..
make
```

Or directly with nvcc:
```bash
nvcc -O2 -std=c++17 -arch=sm_80 topk_gating_kernel.cu -o topk_gating_test
```

## Run

```bash
./topk_gating_test
```

## Tunable Parameters

### 1. BLOCK_SIZE (default: 1024)
- Number of threads per block
- Must be multiple of WARP_SIZE (32)
- Each warp processes one token
- WARPS_PER_BLOCK = BLOCK_SIZE / 32

### 2. MaxNumExperts (32, 64, 96, 128)
- Template parameter for maximum expert count
- Determines register allocation
- Use smallest value >= actual num_experts
- Affects SCORES_PER_THREAD = MaxNumExperts / 32

### 3. MaxNumTopExperts (1, 2, 4, 8)
- Template parameter for K in Top-K
- Must be >= actual topK value
- Affects TopK array size in registers

### 4. DoSoftmaxBeforeTopK
- `true`: Apply softmax to all scores, then select top-K
  - More numerically stable
  - All experts contribute to normalization
- `false`: Select top-K scores, then apply softmax
  - Faster (softmax on K elements instead of N)
  - Only selected experts in normalization

## Algorithm Details

### Warp-Level TopK Reduction
Each warp handles one token:
1. Each thread loads `MaxNumExperts/32` scores
2. Pack score+index into 64-bit value for atomic comparison
3. Use `cg::reduce` with `greater<uint64_t>` for warp max
4. Repeat K times, invalidating previous winners

### Score Packing
```
packed = (twiddled_float << 32) | (65535 - index)
```
- Float twiddling converts to sortable unsigned
- Index negation gives priority to smaller indices (stability)

### Softmax
```
softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
```
- Uses warp reduction for max and sum
- Numerically stable (subtract max before exp)

## Memory Layout

**Input:**
- `routerLogits`: [num_tokens, num_experts] - Float logits from router

**Output:**
- `topkValues`: [num_tokens, topK] - Routing weights (softmax normalized)
- `topkIndices`: [num_tokens, topK] - Selected expert indices

## Performance Characteristics

- **One warp per token**: Good parallelism for large batches
- **Register-heavy**: Scores stored in registers (fast)
- **Warp reduction**: Efficient within-warp communication
- **Compute bound**: Multiple warp reductions per token

## TRT-LLM Specific Features

TRT-LLM version includes:
- Support for half/bfloat16 inputs
- PDL (Programmatic Dependent Launch) for Hopper
- Optimized TopK reduction using CUB utilities
- Fast redux instructions on SM100+

## Supported Configurations

| num_experts | topK | Supported |
|-------------|------|-----------|
| <= 32       | 1-8  | Yes       |
| <= 64       | 1-8  | Yes       |
| <= 96       | 1-8  | Yes       |
| <= 128      | 1-8  | Yes       |

For larger configurations, need to add template instantiations.
