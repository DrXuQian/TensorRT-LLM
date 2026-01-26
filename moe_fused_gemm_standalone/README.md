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
- Added wrapper APIs with FP16 kernel variants + an occupancy-based selector:
  - `fused_moe_gemm_sm80_wrappers.{h,cu}`
- Added a standalone test (`test/test_moe_fused_gemm.cu`) with CLI args, a
  one-time config search (or optional profiling-based selection), basic
  benchmarking knobs, and an optional CPU reference for small cases.
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
- No full tactic/heuristic system outside the fused SM80 list:
  - the default selector follows TRT-LLM's occupancy + wave scoring logic over
    the fused SM80 configs only (no non-fused fallback).
  - optional `--profile` mode benchmarks the fused list and picks the best.
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
  --num_experts=32 --experts_per_token=4 --op=both --warmup=10 --iters=100
```

List the supported SM80 configs:

```
moe_fused_gemm_standalone/build/test_moe_fused_gemm --list_configs
```

Force a specific config (applies to FC1/FC2):

```
moe_fused_gemm_standalone/build/test_moe_fused_gemm \
  --num_tokens=256 --hidden_size=1024 --inter_size=768 \
  --num_experts=32 --experts_per_token=4 --op=both \
  --config=16x128x64x3 --warmup=10 --iters=100
```
The `--config` flag also accepts comma-separated values, e.g. `--config=16,128,64,3`.

Profile all SM80 fused configs and pick the fastest:

```
moe_fused_gemm_standalone/build/test_moe_fused_gemm \
  --num_tokens=3823 --hidden_size=2048 --inter_size=768 \
  --num_experts=128 --experts_per_token=8 --op=both \
  --profile --warmup=10 --iters=200
```

Nsight Compute mode (`--ncu`)
-----------------------------
`--ncu` disables the separate warmup loop and sets a larger default iteration
count (if you didn't pass `--iters=...`). It requires `--config=...` to avoid
any config search/profiling noise.

Example:

```
# 1) Pick a config first (prints the selected config)
moe_fused_gemm_standalone/build/test_moe_fused_gemm --op=fc1

# 2) Rerun with --ncu and the chosen config
moe_fused_gemm_standalone/build/test_moe_fused_gemm --op=fc1 --ncu --iters=2000 --config=16x128x64x3
```

CPU reference check (small shapes only):

```
moe_fused_gemm_standalone/build/test_moe_fused_gemm \
  --num_tokens=16 --hidden_size=128 --inter_size=128 \
  --num_experts=4 --experts_per_token=2 --op=both --verify
```

Debug logging
-------------
To print per-config filtering/scoring details during selection, enable:

- CLI: `--debug`
- or env: `MOE_FUSED_PROFILE_LOG=1`

Traverse all configs (no internal selection)
-------------------------------------------
This helper runs the binary once per config (forced via `--config=...`):

```
moe_fused_gemm_standalone/scripts/run_all_configs.sh --op fc1 --iters 200 --out results_fc1.txt
```

Init + run usage (recommended)
------------------------------
The fused selector runs once to pick configs, then the selected config is
passed into the execution calls:

```
Sm80FusedMoeGemmConfig fc1_cfg{};
Sm80FusedMoeGemmConfig fc2_cfg{};
sm80_fused_moe_select_config_fc1(fc1_cfg, total_tokens_including_expert, num_experts,
    num_rows, inter_size, hidden_size, sm_count);
sm80_fused_moe_select_config_fc2(fc2_cfg, total_tokens_including_expert, num_experts,
    num_rows, hidden_size, inter_size, sm_count);

sm80_fused_moe_fc1_swiglu_fp16_with_config(..., fc1_cfg);
sm80_fused_moe_fc2_identity_fp16_with_config(..., fc2_cfg);
```

Selection method (aligned with TRT-LLM)
--------------------------------------
The selector mirrors TRT-LLM's occupancy + wave scoring heuristic:

- Query per-config occupancy via the fused launcher (`kernel_occupancy` path).
- Estimate total CTAs using `total_tokens_including_expert` per expert:
  `sum_e ceil(rows_e / tile_m) * ceil(gemm_n / tile_n)`.
- Compute waves: `num_waves_total - (ctas / ctas_per_wave)` and pick the lowest
  score; tie-break by fewer waves, then higher stages, then larger `tile_m`.

SM80 fused config list (full)
-----------------------------
Tile shapes and stages match TRT-LLM's SM80 fused generator:

- CTA shapes: (16,128,64), (16,256,64), (32,128,64), (64,128,64), (128,128,64)
- Stages: 2, 3, 4

Notes
-----
- `hidden_size` and `inter_size` must be multiples of 64 for the fused configs.
- The CPU reference only runs for small shapes and is a sanity check, not a full
  TRT-LLM correctness test.
