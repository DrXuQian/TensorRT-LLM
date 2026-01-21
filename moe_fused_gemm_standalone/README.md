Standalone SM80 Fused MoE GEMM (FP16)
====================================

This directory contains a minimal, standalone build of TensorRT-LLM's SM80
fused MoE GEMM kernel. It exposes two FP16 entry points:

- FC1: fused GEMM + SwiGLU (gated)
- FC2: fused GEMM + Identity

What I changed / extracted
--------------------------
- Copied only the fused MoE GEMM kernel stack from TRT-LLM:
  - `cutlass_extensions` headers needed by `fused_moe_kernel*`
  - SM80 fused launcher `fused_moe_gemm_launcher_sm80.{h,inl}`
- Added minimal TRT-LLM stubs:
  - `tensorrt_llm/common/assert.h` (TLLM_CHECK macros)
  - `tensorrt_llm/common/cudaUtils.h` (check_cuda_error)
  - `tensorrt_llm/common/config.h` (namespace macros)
- Added wrapper APIs with fixed FP16 kernel config:
  - `fused_moe_gemm_sm80_wrappers.{h,cu}`
- Added a standalone test (`test/test_moe_fused_gemm.cu`) with CLI args.
- Patched `fused_moe_gemm_launcher_sm80.inl` error formatting to avoid
  `std::string` so it works with the minimal TLLM_CHECK stub.

What is NOT included (limitations)
----------------------------------
- No run_moe or routing kernels:
  - no sorting/permute, expand, finalize, or top-k dispatch
  - inputs must already be expanded and grouped by expert
- No fused-gated-activation toggle:
  - FC1 always uses gated SwiGLU via `EpilogueOpDefaultSilu` and
    `EpilogueRouting(..., true)`
- No non-fused grouped GEMM fallback
- FP16 only:
  - no BF16, FP8, FP4, INT8/INT4, quantized weights, or DeepSeek blockscale
- No SM90/TMA paths
- No kernel heuristic or tactic selection:
  - fixed tile config (128x128x64, stages=2)
- No LoRA, bias scaling, or extra fusion paths (only optional bias pointer)

Input/weight layout assumptions
-------------------------------
- `total_tokens_including_expert` is a length `num_experts` prefix-sum of
  expanded rows per expert.
- Expanded input `A` is grouped by expert in the same order as the prefix-sum.
- FC1 weights `B` are laid out as:
  - `[expert][2][N][K]` where the second dimension is (value, gate).
- FC2 weights `B` are laid out as:
  - `[expert][N][K]`.

Build
-----
From the repo root:

```
cmake -S moe_fused_gemm_standalone -B moe_fused_gemm_standalone/build
cmake --build moe_fused_gemm_standalone/build -j8
```

Run
---
Example (small):

```
moe_fused_gemm_standalone/build/test_moe_fused_gemm \
  --num_tokens=256 --hidden_size=1024 --inter_size=768 \
  --num_experts=32 --experts_per_token=4 --op=both
```

Notes
-----
- The test only does a lightweight checksum; no CPU reference is included.
- This is intended for kernel extraction and benchmarking, not full MoE
  correctness vs TRT-LLM.
