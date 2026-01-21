# MoE Finalize Routing Kernel

## Overview

This kernel is the final step in MoE computation. It unpermutes the expert outputs back to original token order and performs weighted reduction across the k selected experts per token.

**Original source:** `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu`

## What this kernel does

1. For each original token:
   - Gather outputs from k experts (using permutation maps)
   - Multiply each expert output by its routing weight
   - Sum the weighted outputs
   - Optionally add per-expert bias
2. Produces final output of shape `[num_tokens, hidden_size]`

## Build

```bash
mkdir build && cd build
cmake ..
make
```

Or directly with nvcc:
```bash
nvcc -O2 -std=c++17 -arch=sm_80 finalize_routing_kernel.cu -o finalize_routing_test
```

## Run

```bash
./finalize_routing_test
```

## Tunable Parameters

### 1. FINALIZE_THREADS_PER_BLOCK (default: 256)
- Number of threads per block
- Each block handles one output token
- Affects GPU occupancy

### 2. FINALIZE_ELEM_PER_THREAD
- Elements processed per thread per iteration
- For FP16: 128 bits / 16 bits = 8 elements
- For FP32: 128 bits / 32 bits = 4 elements

### 3. ScaleMode
- `DEFAULT`: Use routing weights from softmax
- `NO_SCALE`: All weights = 1.0 (uniform contribution)

## Memory Layout

**Input:**
- `expanded_permuted_rows`: [num_tokens * k, hidden_size] - Expert outputs (permuted)
- `scales`: [num_tokens, k] - Routing weights
- `unpermuted_row_to_permuted_row`: [num_tokens * k] - Reverse permutation
- `token_selected_experts`: [num_tokens, k] - Expert indices

**Output:**
- `reduced_unpermuted_output`: [num_tokens, hidden_size] - Final MoE output

**Optional:**
- `bias`: [num_experts, hidden_size] - Per-expert bias

## Algorithm

```
for each token i:
    output[i] = 0
    for k in 0..experts_per_token:
        expert_id = token_selected_experts[i, k]
        permuted_row = unpermuted_row_to_permuted_row[i + k * num_tokens]
        weight = scales[i, k]
        output[i] += weight * (expanded_permuted_rows[permuted_row] + bias[expert_id])
```

## Performance Characteristics

- **Gather-intensive**: Random reads from expert outputs
- **Reduction**: k-way sum per output element
- **Memory bound**: Limited by random access latency
- **One token per block**: Good parallelism

## Memory Access Pattern

- Input reads: Scattered (following permutation)
- Output writes: Coalesced (sequential)
- Bias reads: Same expert bias reused across column

## TRT-LLM Variants

TRT-LLM has two versions:
1. **finalizeMoeRoutingKernel**: Standard version, fills entire output
2. **finalizeMoeRoutingNoFillingKernel**: Optimized for sparse cases, skips already-processed tokens

## Multi-Node Support

For expert parallelism across nodes:
- `start_expert_id`: First expert ID on this node
- `num_experts_per_node`: Number of local experts
- Experts outside range are skipped
- Requires AllReduce after to combine partial results

## Performance Tips

1. **Maximize occupancy**: Tune FINALIZE_THREADS_PER_BLOCK
2. **Memory coalescing**: Hidden size should be multiple of 128
3. **Reduce k**: Fewer experts per token = faster reduction
4. **Skip unused experts**: Use NO_SCALE mode if weights not needed
