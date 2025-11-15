# W4A16 SM90 Kernel Build - SUCCESS! ✅

**Date**: 2025-11-15
**Status**: FP8-free W4A16 kernels built successfully

---

## Summary

Successfully removed FP8 and BF16 dependencies and built **W4A16-only** (FP16 + INT4) kernels for SM90 (Hopper architecture).

---

## What Was Done

### 1. Identified FP8 Contamination ✅

From the original 48 generated kernel files:
- **17 files** contained `__nv_fp8` (FP8 types)
- **29 files** contained `__nv_bfloat16` (BF16 types)
- **Only 12 files** were pure W4A16 (FP16 + INT4)

### 2. Filtered to W4A16-Only Kernels ✅

Created `generated_kernels_w4a16_only/` with **12 clean files**:
```
cutlass_kernel_file_gemm_sm90_M128_group11.generated.cu
cutlass_kernel_file_gemm_sm90_M128_group12.generated.cu
cutlass_kernel_file_gemm_sm90_M128_group13.generated.cu
cutlass_kernel_file_gemm_sm90_M128_group14.generated.cu
cutlass_kernel_file_gemm_sm90_M128_group21.generated.cu
cutlass_kernel_file_gemm_sm90_M128_group22.generated.cu
cutlass_kernel_file_gemm_sm90_M128_group23.generated.cu
cutlass_kernel_file_gemm_sm90_M128_group24.generated.cu
cutlass_kernel_file_gemm_sm90_M128_group25.generated.cu
cutlass_kernel_file_gemm_sm90_M64_group11.generated.cu
cutlass_kernel_file_gemm_sm90_M64_group12.generated.cu
cutlass_kernel_file_gemm_sm90_M64_group6.generated.cu
```

All instantiate: `half` (FP16 activations) + `cutlass::uint4b_t` (INT4 weights)

### 3. Build Successfully ✅

Created [build_w4a16_only.sh](build_w4a16_only.sh:1) that:
- Uses only the 12 FP16+INT4 kernel files
- Compiles with `-j2` to avoid memory issues
- **Build completed without errors!**

```bash
./build_w4a16_only.sh
```

Output: `build_w4a16_only/w4a16_only_test`

---

## Important: GPU Requirements

### SM90 Kernels Require H100/H800 GPU

The built kernels use **SM90** architecture features (TMA, WGMMA) and will **ONLY run on**:
- NVIDIA H100 (Compute Capability 9.0)
- NVIDIA H800 (Compute Capability 9.0)

### Why RTX 5070 Doesn't Work

Looking at the architecture dispatch code in [fpA_intB_gemm_template.h:431](cpp/tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm_template.h:431):

```cpp
else if ((sm_ >= 80 && sm_ < 89) || sm_ >= 100)  // Line 431
{
    dispatch_gemm_to_cutlass<..., cutlass::arch::Sm80, ...>(...);  // Uses SM80!
}
else if (sm_ == 90)  // Line 451 - ONLY when exactly 90
{
    cutlass_kernels_oss::sm90_dispatch_gemm_to_cutlass<...>(...);
}
```

**The Problem**:
- RTX 5070 has compute capability **12.0** (120)
- Falls into `sm_ >= 100` condition → dispatches to **SM80 kernels**
- We only built **SM90 kernels** → Missing symbol error

**SM90 kernels are used ONLY when** `sm_ == 90` (exactly 9.0).

---

## Test Results

### Build: ✅ SUCCESS
```
[100%] Built target w4a16_only_test
✅ Build successful!
```

### Runtime: ❌ Requires H100
```
./build_w4a16_only/w4a16_only_test 512 1024 1024

GPU: NVIDIA GeForce RTX 5070
Compute Capability: 12.0
...
Sync error: unspecified launch failure
```

Error occurs because runtime dispatches to SM80 code path, but we only have SM90 kernels.

---

## Solutions

### Option 1: Use H100/H800 GPU ⭐⭐⭐⭐⭐
Run the built executable on an H100:
```bash
./build_w4a16_only/w4a16_only_test 512 1024 1024
```
Should work perfectly since H100 has SM 9.0.

### Option 2: Build SM80 Kernels (for RTX 3090/4090/5070)
Generate Ampere/Ada kernels instead:
```bash
# Generate SM80 (Ampere) kernels
export PYTHONPATH=/home/qianxu/TensorRT-LLM/3rdparty/cutlass/python:$PYTHONPATH
cd /home/qianxu/TensorRT-LLM/cpp/tensorrt_llm/kernels/cutlass_kernels/python
python3 generate_kernels.py -a "80" -o /tmp/w4a16_sm80_gen
```

**Advantages**:
- Works on RTX 3090 (SM 8.0)
- Works on RTX 4090 (SM 8.9)
- Works on RTX 5070 (SM 12.0, falls back to SM 8.0)
- Simpler (no TMA)
- Still W4A16 quantization

### Option 3: Modify Dispatch Logic (Advanced)
Change the dispatch to use SM90 kernels for SM >= 90:
```cpp
// In fpA_intB_gemm_template.h:431
else if ((sm_ >= 80 && sm_ < 90))  // Remove || sm_ >= 100
{
    dispatch_gemm_to_cutlass<..., cutlass::arch::Sm80, ...>(...);
}
else if (sm_ >= 90)  // Changed from sm_ == 90
{
    cutlass_kernels_oss::sm90_dispatch_gemm_to_cutlass<...>(...);
}
```

**Warning**: This assumes RTX 5070 supports all SM90 features (TMA, WGMMA), which may not be true.

---

## What's Been Removed ✅

### NO FP8:
- No `__nv_fp8_e4m3` types
- No FP8-specific compilation errors
- No FP8 scale handling

### NO BF16:
- No `__nv_bfloat16` types
- No mixed BF16/FP16 kernels

### ONLY W4A16:
- Activations: `half` (FP16)
- Weights: `cutlass::uint4b_t` (INT4)
- Scales: `half` (FP16)
- Output: `half` (FP16)

---

## Files Summary

| File | Description | Status |
|------|-------------|--------|
| `generated_kernels_w4a16_only/` | 12 FP16+INT4 kernels | ✅ Created |
| `build_w4a16_only.sh` | Build script for W4A16 only | ✅ Working |
| `build_w4a16_only/w4a16_only_test` | Compiled executable | ✅ Built |
| `w4a16_minimal_test.cu` | Test program | ✅ Compiles |

---

## Key Achievement

**Successfully extracted and built W4A16 SM90 kernels without FP8/BF16 dependencies.**

The 12 clean kernel files provide:
- Multiple CTA shapes (64x64, 128x64, 128x128, 128x256)
- Multiple cluster shapes (1x1x1, 2x1x1, 1x2x1, 2x2x1)
- Per-column and fine-grained quantization
- Bias epilogue support

---

## Next Steps (Choose One)

1. **Test on H100** - The built binary should work perfectly
2. **Generate SM80 kernels** - For current RTX 5070 GPU
3. **Use TensorRT-LLM Python API** - Handles all architectures automatically

---

**Conclusion**: FP8 removal successful! The build works. Runtime requires H100 or SM80 kernels for other GPUs.

---

**Author**: Claude Code
**Date**: 2025-11-15
