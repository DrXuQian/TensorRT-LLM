# MoE SwiGLU/GeGLU Activation Kernel

## Overview

This kernel applies gated activation functions (SwiGLU or GeGLU) to the FC1 GEMM output in MoE FFN blocks. It's a fused element-wise operation that combines the gate and linear paths.

**Original source:** `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu`

## What this kernel does

**SwiGLU (default):**
```
output = SiLU(gate) * linear
SiLU(x) = x * sigmoid(x)
```

**GeGLU:**
```
output = GELU(gate) * linear
```

**SwigluBias (with per-expert scaling):**
```
gate_act = alpha * SiLU(gate) + beta
gate_act = clamp(gate_act, -limit, limit)
output = gate_act * linear
```

## Memory Layout

**Input (FC1 output):** `[num_tokens, inter_size * 2]`
- First `inter_size` columns: linear values
- Second `inter_size` columns: gate values

**Output:** `[num_tokens, inter_size]`

## Build

```bash
mkdir build && cd build
cmake ..
make
```

Or directly with nvcc:
```bash
nvcc -O2 -std=c++17 -arch=sm_80 swiglu_activation_kernel.cu -o swiglu_activation_test
```

## Run

```bash
./swiglu_activation_test
```

## Tunable Parameters

### 1. ACTIVATION_THREADS_PER_BLOCK (default: 256)
- Number of threads per block
- Each block handles one token (row)
- Affects GPU occupancy

### 2. ELEM_PER_THREAD (default: 8 for FP16)
- Elements processed per thread per iteration
- Uses 128-bit vectorized loads/stores
- For FP16: 128 bits / 16 bits = 8 elements

### 3. Activation Type
- `Swiglu`: SiLU-gated linear unit
- `Geglu`: GELU-gated linear unit
- `SwigluBias`: SwiGLU with per-expert alpha, beta, limit

### 4. Per-Expert Parameters (SwigluBias only)
- `swiglu_alpha[num_experts]`: Scaling factor
- `swiglu_beta[num_experts]`: Bias term
- `swiglu_limit[num_experts]`: Clamping value

## Performance Characteristics

- **Compute bound**: Activation functions (exp, tanh) are compute intensive
- **Memory access**: Reads 2x more than writes (gate + linear -> output)
- **One token per block**: Good parallelism for large batch sizes
- **Vectorized access**: 128-bit loads/stores for memory efficiency

## TRT-LLM Integration Notes

In TRT-LLM, this kernel can be:
1. **Fused into GEMM epilogue**: For SM80 with certain tile sizes
2. **Run as separate kernel**: When fusion isn't possible or for Hopper TMA

Fusion decision based on:
- `MoeGemmRunner::supportsFusedGatedActivation()`
- Tile dimensions must match GEMM output
- Not supported for TMA warp-specialized GEMMs

## Comparison with Fused GEMM+Activation

| Approach | Pros | Cons |
|----------|------|------|
| Separate kernel | Simple, always works | Extra memory read/write |
| Fused epilogue | Better memory efficiency | Limited tile sizes |
| TMA-based | Hopper optimization | More complex |

For SM80 with standard configurations, the fused approach is typically used.
