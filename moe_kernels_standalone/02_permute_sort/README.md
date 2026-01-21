# MoE Permute/Sort Kernel

## Overview

This kernel sorts tokens by their selected expert IDs and builds permutation maps for MoE computation. It's the second step in the MoE pipeline after the TopK gating.

**Original source:** `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu`

## What this kernel does

1. Takes token-expert assignments from the router
2. Sorts tokens so that tokens assigned to the same expert are contiguous
3. Builds bidirectional permutation maps:
   - `permuted_row_to_unpermuted_row`: Maps sorted position to original position
   - `unpermuted_row_to_permuted_row`: Maps original position to sorted position
4. Computes `expert_first_token_offset`: Starting position for each expert's tokens

## Build

```bash
mkdir build && cd build
cmake ..
make
```

Or directly with nvcc:
```bash
nvcc -O2 -std=c++17 -arch=sm_80 permute_sort_kernel.cu -o permute_sort_test
```

## Run

```bash
./permute_sort_test
```

## Tunable Parameters

### 1. BLOCK_SIZE (32, 64, 128, 256)
- Number of threads per block
- Must be power of 2
- Fused kernel limited to `num_tokens <= BLOCK_SIZE` (max 256)
- Larger block size handles more tokens in single kernel launch

### 2. LOG2_NUM_EXPERTS
- Number of bits for radix sort = log2(num_experts + 2)
- For 128 experts: 8 bits (log2(130) = 7.02, rounded up to 8)
- For 256 experts: 9 bits
- Maximum supported: 9 (up to 510 experts)

### 3. EXPERTS_PER_TOKEN
- Number of experts selected per token (top-k value)
- Supported values in TRT-LLM: 1, 2, 4, 6, 8
- Affects unrolling and register usage

## Implementation Details

### Fused Kernel (num_tokens <= 256)
Uses CUB `BlockRadixRank` for efficient in-block sorting:
- Single kernel launch
- No global memory for intermediate results
- Shared memory for CUB temporary storage

### Three-Step Approach (num_tokens > 256)
1. **Block-level counting**: Each block counts tokens per expert using atomics in shared memory
2. **Global prefix sum**: Compute exclusive prefix sum across blocks for each expert
3. **Merge and write**: Write final permutation using block-local offsets

## Memory Layout

**Input:**
- `token_selected_experts`: [num_tokens, experts_per_token]

**Output:**
- `permuted_row_to_unpermuted_row`: [num_tokens * experts_per_token]
- `unpermuted_row_to_permuted_row`: [num_tokens * experts_per_token]
- `expert_first_token_offset`: [num_experts + 1]

## Performance Considerations

- Fused kernel is preferred for small batches (decode phase)
- Three-step approach handles large batches (prefill phase)
- Shared memory usage scales with num_experts
- Radix sort is O(n) complexity

## TRT-LLM Configuration

In TRT-LLM, this kernel's behavior is controlled by:
- `MOEParallelismConfig`: Determines which experts are local to this node
- `start_expert`, `end_expert`: Expert range for multi-node parallelism
