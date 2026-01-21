# MoE Kernels Standalone

This directory contains standalone versions of the MoE (Mixture of Experts) kernels extracted from TensorRT-LLM. Each kernel can be compiled and tested independently.

## Source Files in TensorRT-LLM

| Kernel | Original Source File |
|--------|---------------------|
| 01_topk_gating | `cpp/tensorrt_llm/kernels/customMoeRoutingKernels.cu` |
| 02_permute_sort | `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu` |
| 03_expand_input | `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu` |
| 04_swiglu_activation | `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu` |
| 05_finalize_routing | `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu` |

Additional related files:
- `cpp/tensorrt_llm/kernels/moeTopKFuncs.cuh` - TopK reduction utilities
- `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/fused_moe_gemm_launcher_sm80.inl` - SM80 GEMM launcher

## MoE-SwiGLU-FFN Pipeline Overview

The Mixture of Experts Feed-Forward Network with SwiGLU activation processes input through the following stages:

```
                        MoE-SwiGLU-FFN Pipeline
                        ======================

Input: X [num_tokens, hidden_size]
        │
        │ Router: X @ W_router → logits [num_tokens, num_experts]
        ▼
┌───────────────────────────────────────────────────────────────────┐
│ STEP 1: TopK Gating (01_topk_gating)                              │
│ ─────────────────────────────────────                             │
│ Input:  router_logits [num_tokens, num_experts]                   │
│ Output: expert_indices [num_tokens, k]                            │
│         routing_weights [num_tokens, k]  (softmax normalized)     │
│                                                                   │
│ Algorithm:                                                        │
│   1. For each token, find top-k experts with highest logits       │
│   2. Apply softmax to selected logits → routing weights           │
│   3. Weights sum to 1.0 per token                                 │
└───────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│ STEP 2: Permute/Sort (02_permute_sort)                            │
│ ─────────────────────────────────────                             │
│ Input:  expert_indices [num_tokens, k]                            │
│ Output: permuted_row_to_unpermuted_row [num_tokens * k]           │
│         unpermuted_row_to_permuted_row [num_tokens * k]           │
│         expert_first_token_offset [num_experts + 1]               │
│                                                                   │
│ Algorithm:                                                        │
│   1. Sort all (token, expert) pairs by expert_id                  │
│   2. Build bidirectional permutation maps                         │
│   3. Compute offset where each expert's tokens start              │
│                                                                   │
│ Purpose: Group tokens by expert for efficient batched GEMM        │
│                                                                   │
│ Before sorting:     After sorting (grouped by expert):            │
│ Token0→[E2,E5]      Expert0: [Token1, Token3, ...]                │
│ Token1→[E0,E3]      Expert1: [Token2, Token5, ...]                │
│ Token2→[E1,E0]      Expert2: [Token0, Token4, ...]                │
│ ...                 ...                                           │
└───────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│ STEP 3: Expand Input (03_expand_input)                            │
│ ─────────────────────────────────────                             │
│ Input:  X [num_tokens, hidden_size]                               │
│         permuted_row_to_unpermuted_row [num_tokens * k]           │
│ Output: X_expanded [num_tokens * k, hidden_size]                  │
│         permuted_scales [num_tokens * k]                          │
│                                                                   │
│ Algorithm:                                                        │
│   1. For each permuted row, look up source token                  │
│   2. Copy input row to permuted position                          │
│   3. Reorder routing weights accordingly                          │
│                                                                   │
│ Purpose: Replicate each token k times, reordered by expert        │
│                                                                   │
│ Memory layout transformation:                                     │
│ X[token] → X_expanded[permuted_positions_for_token's_experts]     │
└───────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│ STEP 4: FC1 GEMM (Grouped GEMM - not in this repo)                │
│ ─────────────────────────────────────                             │
│ Input:  X_expanded [num_tokens * k, hidden_size]                  │
│         W_fc1 [num_experts, hidden_size, inter_size * 2]          │
│ Output: FC1_out [num_tokens * k, inter_size * 2]                  │
│                                                                   │
│ Computation: For each expert e:                                   │
│   tokens_for_e = X_expanded[expert_offset[e]:expert_offset[e+1]]  │
│   FC1_out[...] = tokens_for_e @ W_fc1[e]                          │
│                                                                   │
│ Note: Output has 2x inter_size for gated activation               │
│       First half = linear path, Second half = gate path           │
└───────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│ STEP 5: SwiGLU Activation (04_swiglu_activation)                  │
│ ─────────────────────────────────────────────────                 │
│ Input:  FC1_out [num_tokens * k, inter_size * 2]                  │
│ Output: act_out [num_tokens * k, inter_size]                      │
│                                                                   │
│ Algorithm:                                                        │
│   linear = FC1_out[:, :inter_size]                                │
│   gate   = FC1_out[:, inter_size:]                                │
│   act_out = SiLU(gate) * linear                                   │
│                                                                   │
│ Where SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))                │
│                                                                   │
│ Variants:                                                         │
│   - SwiGLU: SiLU(gate) * linear                                   │
│   - GeGLU:  GELU(gate) * linear                                   │
│   - SwiGLUBias: (alpha * SiLU(gate) + beta) * linear, clamped     │
└───────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│ STEP 6: FC2 GEMM (Grouped GEMM - not in this repo)                │
│ ─────────────────────────────────────                             │
│ Input:  act_out [num_tokens * k, inter_size]                      │
│         W_fc2 [num_experts, inter_size, hidden_size]              │
│ Output: FC2_out [num_tokens * k, hidden_size]                     │
│                                                                   │
│ Computation: For each expert e:                                   │
│   tokens_for_e = act_out[expert_offset[e]:expert_offset[e+1]]     │
│   FC2_out[...] = tokens_for_e @ W_fc2[e]                          │
└───────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│ STEP 7: Finalize Routing (05_finalize_routing)                    │
│ ─────────────────────────────────────────────                     │
│ Input:  FC2_out [num_tokens * k, hidden_size]                     │
│         routing_weights [num_tokens, k]                           │
│         unpermuted_row_to_permuted_row [num_tokens * k]           │
│ Output: Y [num_tokens, hidden_size]                               │
│                                                                   │
│ Algorithm:                                                        │
│   For each original token t:                                      │
│     Y[t] = 0                                                      │
│     For each selected expert e in token t's top-k:                │
│       permuted_row = unpermuted_row_to_permuted_row[t, e]         │
│       Y[t] += routing_weights[t, e] * FC2_out[permuted_row]       │
│                                                                   │
│ Purpose: Unpermute expert outputs and compute weighted sum        │
└───────────────────────────────────────────────────────────────────┘
        │
        ▼
Output: Y [num_tokens, hidden_size]
```

## Kernel Details

### 01_topk_gating
**Function**: Select top-k experts for each token and compute routing weights

**Key Features**:
- Warp-level parallel TopK using cooperative groups
- Efficient value+index packing for atomic max operations
- Softmax normalization (before or after TopK selection)

**Original Functions**: `customMoeRoutingKernel`

---

### 02_permute_sort
**Function**: Sort tokens by expert assignment and build permutation maps

**Key Features**:
- Fused kernel for small token counts (≤256) using CUB BlockRadixRank
- Three-step approach for large token counts (block count → global prefix sum → merge)
- Builds bidirectional index mappings

**Original Functions**: `fusedBuildExpertMapsSortFirstTokenKernel`, `blockExpertPrefixSum`, `globalExpertPrefixSum`, `mergeExpertPrefixSum`

---

### 03_expand_input
**Function**: Duplicate and reorder input rows for grouped GEMM

**Key Features**:
- Vectorized 128-bit memory operations
- Supports FP16/FP32 data types
- Permutes routing weights alongside activations

**Original Functions**: `expandInputRowsKernel`

---

### 04_swiglu_activation
**Function**: Apply gated activation (SwiGLU/GeGLU) to FC1 output

**Key Features**:
- Fuses gate and linear paths
- Vectorized computation (8 FP16 elements per thread)
- Optional per-expert alpha/beta/limit parameters

**Original Functions**: `doGatedActivationKernel`

---

### 05_finalize_routing
**Function**: Unpermute expert outputs and compute weighted reduction

**Key Features**:
- Gathers outputs from k experts per token
- Applies routing weights for weighted sum
- Optional bias addition

**Original Functions**: `finalizeMoeRoutingKernel`

---

## Directory Structure

```
moe_kernels_standalone/
├── README.md                 # This file
├── 01_topk_gating/
│   ├── topk_gating_kernel.cu
│   ├── CMakeLists.txt
│   └── README.md
├── 02_permute_sort/
│   ├── permute_sort_kernel.cu
│   ├── CMakeLists.txt
│   └── README.md
├── 03_expand_input/
│   ├── expand_input_kernel.cu
│   ├── CMakeLists.txt
│   └── README.md
├── 04_swiglu_activation/
│   ├── swiglu_activation_kernel.cu
│   ├── CMakeLists.txt
│   └── README.md
└── 05_finalize_routing/
    ├── finalize_routing_kernel.cu
    ├── CMakeLists.txt
    └── README.md
```

## Quick Start

### Build all kernels:
```bash
cd moe_kernels_standalone

for dir in 0*/; do
    echo "Building ${dir}..."
    cd $dir
    mkdir -p build && cd build
    cmake ..
    make
    cd ../..
done
```

### Run all tests:
```bash
for dir in 0*/build/; do
    echo "=== Testing ${dir} ==="
    ./${dir}*_test
    echo ""
done
```

## Test Parameters

Based on user's test case:
- `num_tokens`: 3823
- `hidden_size`: 2048
- `inter_size`: 768
- `num_experts`: 128
- `experts_per_token`: 8

## Tunable Parameters Summary

| Kernel | Parameter | Values | Description |
|--------|-----------|--------|-------------|
| TopK | BLOCK_SIZE | 1024 | Threads per block |
| TopK | MaxNumExperts | 32,64,96,128 | Max expert count (template) |
| TopK | MaxNumTopExperts | 1,2,4,8 | Max K value (template) |
| TopK | DoSoftmaxBeforeTopK | true/false | Softmax timing |
| Permute/Sort | BLOCK_SIZE | 32-256 | Threads per block |
| Permute/Sort | LOG2_NUM_EXPERTS | 5-9 | Radix sort bits |
| Permute/Sort | EXPERTS_PER_TOKEN | 1,2,4,6,8 | K value (template) |
| Expand | THREADS_PER_BLOCK | 256 | Threads per block |
| Expand | ELEM_PER_THREAD | 8 (FP16) | Vector width |
| SwiGLU | THREADS_PER_BLOCK | 256 | Threads per block |
| SwiGLU | ActivationType | Swiglu/Geglu/SwigluBias | Activation variant |
| Finalize | THREADS_PER_BLOCK | 256 | Threads per block |
| Finalize | ScaleMode | DEFAULT/NO_SCALE | Use routing weights |

## GEMM Kernels (Not Included)

The grouped GEMM kernels (FC1 and FC2) are not included because they have heavy CUTLASS template dependencies. In TensorRT-LLM:

- **SM80 (Ampere)**: Uses CUTLASS 2.x grouped GEMM with warp-level MMA
  - CTA shapes: 32x128x64, 64x128x64, 128x128x64
  - Stages: 2, 3, 4
  - Supports fused SwiGLU epilogue

- **SM90 (Hopper)**: Uses CUTLASS 3.x TMA warp-specialized kernels
  - Different tile configurations
  - Programmatic Dependent Launch (PDL) support

## Kernel Fusion Strategies in TensorRT-LLM

TensorRT-LLM implements two major fusion optimizations for MoE kernels. The decision logic is based on GPU architecture, data types, and problem dimensions.

### 1. SwiGLU Fusion (FC1 GEMM + Activation)

SwiGLU activation can be fused into FC1 GEMM epilogue on SM80 (Ampere) architecture.

**Decision Logic** (from `moe_gemm_template_dispatch.h:652-666`):

```cpp
bool supportsFusedGatedActivation(ActivationType activation_type, int gemm_n, int gemm_k) const
{
    return (activation_type == ActivationType::Swiglu || activation_type == ActivationType::Geglu)
        && std::is_same_v<T, WeightType>     // Input and weight types match (non-quantized)
        && !std::is_same_v<T, float>         // Not FP32
        && !use_fp8                           // Not FP8 quantization
        && (this->getSM() >= 80)              // SM80+
        && (gemm_k % 64 == 0)                 // K dimension aligned to 64
        && (gemm_n % 64 == 0)                 // N dimension aligned to 64
        && ENABLE_FUSED_GATED_ACTIVATION;     // Compile-time switch
}

bool isFusedGatedActivation(CutlassGemmConfig gemm_config, ...) const
{
    return supportsFusedGatedActivation(...)
        && !gemm_config.is_tma_warp_specialized;  // NOT TMA Warp Specialized
}
```

**Fusion Decision Flow**:

```
                    FC1 GEMM + SwiGLU Decision
                            |
                            v
                ┌─────────────────────────┐
                │   is_gated_activation?   │
                │ (Swiglu/Geglu/SwigluBias)│
                └───────────┬─────────────┘
                            │ Yes
                            v
                ┌─────────────────────────┐
                │   SM90+ TMA Warp Spec?   │
                └───────────┬─────────────┘
                      │           │
                     Yes          No
                      │           │
                      v           v
        ┌──────────────────┐   ┌─────────────────────────┐
        │ Separate         │   │ Check supportsFusedGated│
        │ doGatedActivation│   └───────────┬─────────────┘
        │ kernel           │              │
        │                  │        ┌─────┴─────┐
        │ (FC1 outputs     │      Yes         No
        │  inter_size*2)   │        │           │
        └──────────────────┘        v           v
                            ┌──────────────┐  ┌──────────────────┐
                            │ FC1 Epilogue │  │ Separate         │
                            │ fuses SwiGLU │  │ doGatedActivation│
                            │              │  │ kernel           │
                            │(FC1 outputs  │  │(FC1 outputs      │
                            │ inter_size)  │  │ inter_size*2)    │
                            └──────────────┘  └──────────────────┘
```

**SwiGLU Fusion Support Conditions**:

| Condition | Required Value | Notes |
|-----------|---------------|-------|
| Activation Type | Swiglu or Geglu | SwigluBias not supported for fusion |
| Data Type | FP16/BF16 | FP32 not supported |
| Weight Type | Same as activation | Quantized weights (INT4/INT8) not supported |
| FP8 | Not supported | Uses separate path |
| N dimension | % 64 == 0 | GEMM tile alignment |
| K dimension | % 64 == 0 | GEMM tile alignment |
| Architecture | SM80-SM89 | SM90 uses TMA, does not use this fusion |

---

### 2. Finalize Fusion (FC2 GEMM + Reduce)

FC2 GEMM can fuse the finalize routing (weighted reduction) into its epilogue on SM90+ (Hopper) architecture.

**Decision Logic** (from `moe_kernels.h:827-831` and `moe_kernels.cu:4058-4068`):

```cpp
// Check if fusion is possible
bool mayHaveFinalizeFused() const {
    return moe_gemm_runner_.supportsTmaWarpSpecialized()  // TMA support available
        && moe_gemm_runner_.getSM() >= 90                  // SM90+ only (Hopper)
        && use_fused_finalize_                             // Config enabled
        && !use_wfp4a16;                                   // Not W4A16 quantization
}

// Actual usage decision
bool using_fused_finalize = use_fused_finalize_
    && gemm2_using_finalize_fusion      // Config has FINALIZE epilogue type
    && !use_wfp4a16                     // Not W4A16 quantization
    && !use_lora;                       // Not using LoRA
```

**Finalize Fusion Decision Flow**:

```
                    FC2 GEMM + Finalize Decision
                            |
                            v
                ┌─────────────────────────┐
                │      SM90+ (Hopper)?     │
                └───────────┬─────────────┘
                      │           │
                     Yes          No
                      │           │
                      v           v
        ┌─────────────────────┐  ┌──────────────────┐
        │ TMA Warp Specialized│  │ Separate         │
        │ available?          │  │ finalizeMoeRouting│
        └───────────┬─────────┘  │ kernel           │
                    │            └──────────────────┘
              ┌─────┴─────┐
             Yes          No
              │           │
              v           v
    ┌─────────────────┐  ┌──────────────────┐
    │Check conditions:│  │ Separate         │
    │- use_fused_     │  │ finalizeMoeRouting│
    │  finalize=true  │  │ kernel           │
    │- !use_wfp4a16   │  └──────────────────┘
    │- !use_lora      │
    │- GEMM2 config   │
    │  has FINALIZE   │
    │  epilogue type  │
    └───────┬─────────┘
      ┌─────┴─────┐
     All          Any
     Pass         Fail
      │           │
      v           v
┌──────────────┐  ┌──────────────────┐
│ FC2 Epilogue │  │ Separate         │
│ fuses reduce │  │ finalizeMoeRouting│
│              │  │ kernel           │
│ Y[t] = Σ w*x │  └──────────────────┘
└──────────────┘
```

**Finalize Fusion Support Conditions**:

| Condition | Required Value | Notes |
|-----------|---------------|-------|
| Architecture | SM90+ (Hopper) | TMA Warp Specialized required |
| TMA Support | Must be available | `supportsTmaWarpSpecialized()` returns true |
| use_fused_finalize | true | Configuration flag |
| Quantization | Not W4A16 | W4A16 quantization disables fusion |
| LoRA | Not used | LoRA disables fusion |
| GEMM Config | FINALIZE epilogue type | `epilogue_fusion_type == FINALIZE` |

---

### Summary: Fusion Strategy by Architecture

| Architecture | SwiGLU Fusion | Finalize Fusion | Notes |
|--------------|---------------|-----------------|-------|
| SM80 (Ampere) | FC1 Epilogue | Not supported | SwiGLU fused, finalize separate |
| SM89 (Ada) | FC1 Epilogue | Not supported | Same as SM80 |
| SM90 (Hopper) | Separate kernel | FC2 Epilogue | Finalize fused, SwiGLU separate |
| SM90+ with W4A16 | Separate kernel | Separate kernel | Neither fusion supported |

**Key Insight**: SM80 fuses SwiGLU into FC1, while SM90 fuses Finalize into FC2. This is because:
- SM80 uses traditional GEMM with epilogue fusion for activation
- SM90 uses TMA Warp Specialized GEMM which has different epilogue capabilities optimized for reduction

## License

These kernels are extracted from TensorRT-LLM which is licensed under Apache 2.0.
See the original project for full license details.
