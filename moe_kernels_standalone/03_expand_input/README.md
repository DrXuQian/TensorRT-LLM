# MoE Expand Input Rows Kernel

## Overview

This kernel duplicates and permutes input activation rows based on expert routing. Each input token gets replicated to `experts_per_token` rows, placed according to the permutation computed by the sorting kernel.

**Original source:** `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu`

## What this kernel does

1. Takes original input activations of shape `[num_tokens, hidden_size]`
2. Expands to `[num_tokens * experts_per_token, hidden_size]`
3. Reorders rows according to `permuted_row_to_unpermuted_row` mapping
4. Optionally permutes routing weights/scales
5. Groups tokens by assigned expert for efficient GEMM

## Build

```bash
mkdir build && cd build
cmake ..
make
```

Or directly with nvcc:
```bash
nvcc -O2 -std=c++17 -arch=sm_80 expand_input_kernel.cu -o expand_input_test
```

## Run

```bash
./expand_input_test
```

## Tunable Parameters

### 1. EXPAND_THREADS_PER_BLOCK (default: 256)
- Number of threads per block
- Each thread handles 128 bits per iteration (8 FP16 elements)
- Affects GPU occupancy and memory throughput

### 2. ELEM_PER_THREAD
- Elements processed per thread per iteration
- Calculated based on data type:
  - FP16: 128 bits / 16 bits = 8 elements
  - FP32: 128 bits / 32 bits = 4 elements
  - FP8: 128 bits / 8 bits = 16 elements
  - FP4: 128 bits / 4 bits = 32 elements (with block scaling)

### 3. Grid Size
- Automatically calculated based on number of SMs
- Uses `multiProcessorCount * 4` as heuristic
- Each block handles one or more output rows

## Memory Layout

**Input:**
- `unpermuted_input`: [num_tokens, hidden_size] - Original activations
- `unpermuted_scales`: [num_tokens, experts_per_token] - Routing weights
- `permuted_row_to_unpermuted_row`: [num_tokens * experts_per_token] - Index mapping

**Output:**
- `permuted_output`: [num_tokens * experts_per_token, hidden_size] - Expanded activations
- `permuted_scales`: [num_tokens * experts_per_token] - Reordered routing weights

## Memory Access Pattern

- Coalesced reads from source rows (128-bit aligned)
- Coalesced writes to destination rows
- Memory-bound kernel with simple gather pattern
- Effective bandwidth close to memory bandwidth

## TRT-LLM Specific Features

In TRT-LLM, this kernel also supports:
- **FP4/FP8 quantization**: On-the-fly quantization during expansion
- **Block scaling factors**: For MXFP8 and NVFP4 formats
- **AWQ pre-quantization**: Per-expert activation scaling
- **PDL (Programmatic Dependent Launch)**: For Hopper architecture overlap

## Performance Considerations

- Pure memory copy with gather operation
- Bandwidth limited, not compute limited
- Performance scales linearly with hidden_size
- Use pinned memory for host-device transfers if needed
