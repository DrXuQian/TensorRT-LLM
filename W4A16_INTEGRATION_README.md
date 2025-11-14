# W4A16 SM90 Kernel Integration with TensorRT-LLM

**Date**: 2025-11-14
**Status**: Integration Complete - Ready for Testing

---

## Overview

This document describes the integration of standalone W4A16 (4-bit weights, 16-bit activations) Hopper kernel tests into the TensorRT-LLM codebase.

### Key Insight: Use TensorRT-LLM's API, Don't Call Kernels Directly

The previous standalone approach in `/home/qianxu/trt_llm_w4a16_hopper/` called the low-level launcher functions directly, which caused segmentation faults. This was the **wrong approach**.

The **correct approach** is to use TensorRT-LLM's `CutlassFpAIntBGemmRunner` API, which:
- Handles proper initialization of CUTLASS templates
- Configures TMA descriptors correctly
- Manages workspace allocation
- Selects appropriate kernel configurations

---

## Files Created

### 1. Standalone Test (Root Directory)

**File**: [w4a16_sm90_standalone_test.cu](w4a16_sm90_standalone_test.cu)
- Uses `CutlassFpAIntBGemmRunner` API
- Can be built independently from TensorRT-LLM
- Accepts M, N, K dimensions as command-line arguments
- Generates random input data for testing

**Build Files**:
- [CMakeLists_w4a16_test.txt](CMakeLists_w4a16_test.txt) - CMake configuration
- [build_w4a16_test.sh](build_w4a16_test.sh) - Build script

**Build & Run**:
```bash
cd /home/qianxu/TensorRT-LLM
./build_w4a16_test.sh

# Run on H800
cd build_w4a16_test
./w4a16_test [M] [N] [K]

# Example
./w4a16_test 1024 2048 2048
```

### 2. GTest Unit Test (Integrated)

**File**: [cpp/tests/unit_tests/kernels/weightOnly/w4a16_sm90_simple_test.cu](cpp/tests/unit_tests/kernels/weightOnly/w4a16_sm90_simple_test.cu)
- Uses Google Test framework
- Integrated into TensorRT-LLM's test suite
- Automatically skips if GPU is not SM90+

**Modified**: [cpp/tests/unit_tests/kernels/CMakeLists.txt](cpp/tests/unit_tests/kernels/CMakeLists.txt)
- Added: `add_gtest(w4a16SM90SimpleTest weightOnly/w4a16_sm90_simple_test.cu)`

**Build & Run** (when building full TensorRT-LLM):
```bash
# Build TensorRT-LLM (follow official build instructions)
# Then run the specific test:
./build/cpp/tests/unit_tests/kernels/w4a16SM90SimpleTest
```

---

## How It Works

### Old (Wrong) Approach ❌

```cpp
// Directly calling low-level launcher functions
extern "C" void w4a16_sm90_gemm_128(
    half const* A, cutlass::uint4b_t const* B, ...
);

// This causes segfault because:
// - Missing proper CUTLASS initialization
// - TMA descriptors not configured correctly
// - No workspace management
w4a16_sm90_gemm_128(d_A, d_B, ...);  // CRASHES!
```

### New (Correct) Approach ✅

```cpp
// Use TensorRT-LLM's CutlassFpAIntBGemmRunner API
CutlassFpAIntBGemmRunner<half, cutlass::uint4b_t,
    cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> runner;

// Get workspace size
size_t workspace_bytes = runner.getWorkspaceSize(M, N, K);
cudaMalloc(&workspace, workspace_bytes);

// Get available configs
auto configs = runner.getConfigs();
CutlassGemmConfig config = configs[0];

// Call GEMM properly through the API
runner.gemm(
    d_A,                  // activations
    d_B,                  // INT4 weights
    d_scales,             // quantization scales
    nullptr,              // zero points (optional)
    nullptr,              // biases (optional)
    1.0f,                 // alpha
    d_C,                  // output
    M, N, K,              // dimensions
    group_size,           // quantization group size
    config,               // kernel configuration
    workspace,            // workspace
    workspace_bytes,      // workspace size
    stream                // CUDA stream
);
```

---

## Technical Details

### Quantization
- **W4A16**: 4-bit integer weights, 16-bit floating point activations
- **Fine-grained**: Group-wise quantization with configurable group size (default: 128)
- **INT4 Packing**: 2 values per byte

### Kernel Variants
1. **128×128×128 CTA** with TMA Warp Specialized Cooperative
2. **64×128×128 CTA** with TMA Warp Specialized Pingpong

### Memory Layout
- **A (activations)**: Row-major FP16
- **B (weights)**: Special CUTLASS mixed GEMM layout (INT4, preprocessed)
- **Scales**: Per-group FP16 scales
- **C (output)**: Row-major FP16

### Requirements
- **GPU**: Hopper (SM90+), e.g., H100, H800
- **CUDA**: 12.x
- **CMake**: 3.18+
- **Compiler**: C++17

---

## What Changed from Standalone Version

| Aspect | Standalone (`/home/qianxu/trt_llm_w4a16_hopper/`) | Integrated |
|--------|---------------------------------------------------|------------|
| API | Direct launcher calls | `CutlassFpAIntBGemmRunner` |
| Initialization | Manual | Automatic via runner |
| Workspace | Fixed 16MB | Dynamic via `getWorkspaceSize()` |
| Config | Hardcoded | Selected from `getConfigs()` |
| Result | Segfault | ✅ Should work |

---

## Testing on H800

The standalone test should work on H800 because:
1. H800 is SM90 (Hopper) - **supports TMA** ✅
2. We use TensorRT-LLM's official API
3. Proper initialization and configuration
4. Workspace allocation handled correctly

### Expected Output

```
=== W4A16 SM90 Standalone Test using TensorRT-LLM API ===

GPU: NVIDIA H800
Compute Capability: 9.0

Config: M=1024, N=2048, K=2048, group_size=128

Allocating device memory...
Initializing data...

=== Creating CutlassFpAIntBGemmRunner (TensorRT-LLM API) ===
Runner created successfully!
Workspace size: 8388608 bytes (8.00 MB)
Workspace allocated
Available configs: 2
Using config #0: CTA=128x128x128, Cluster=1x1x1, Stages=4

=== Launching GEMM using TensorRT-LLM API ===
GEMM call returned successfully!
No kernel launch errors
Synchronizing stream...
✅ Stream synchronized successfully!

=== Checking Results ===
First 10 outputs:
  C[0] = 123.456789
  C[1] = 234.567890
  ...
  C[9] = 345.678901

Statistics (first 100 elements):
  Zero: 0
  Non-zero: 100

✅ SUCCESS: Kernel executed and produced non-zero outputs!
```

---

## Differences from Old Approach

### Previous Issues (Standalone in `/home/qianxu/trt_llm_w4a16_hopper/`)

1. **Direct Launcher Calls**: Called `w4a16_sm90_gemm_128()` directly
2. **Manual Configuration**: Created wrapper functions with hardcoded templates
3. **Fixed Workspace**: Always allocated 16MB
4. **No Config Selection**: Used hardcoded CTA and cluster shapes
5. **Result**: Segmentation fault during `cudaStreamSynchronize()`

### Current Approach (Integrated)

1. **API Usage**: Uses `CutlassFpAIntBGemmRunner::gemm()`
2. **Automatic Configuration**: Runner handles all template instantiation
3. **Dynamic Workspace**: Calls `getWorkspaceSize()` first
4. **Config Selection**: Queries available configs via `getConfigs()`
5. **Expected Result**: Should work properly ✅

---

## Next Steps

1. **Build the standalone test**:
   ```bash
   cd /home/qianxu/TensorRT-LLM
   ./build_w4a16_test.sh
   ```

2. **Run on H800**:
   ```bash
   cd build_w4a16_test
   ./w4a16_test 1024 2048 2048
   ```

3. **Verify output**:
   - Should see non-zero values in output
   - No segmentation fault
   - Kernel executes successfully

4. **If successful**, commit to git:
   ```bash
   git add w4a16_sm90_standalone_test.cu
   git add CMakeLists_w4a16_test.txt
   git add build_w4a16_test.sh
   git add cpp/tests/unit_tests/kernels/weightOnly/w4a16_sm90_simple_test.cu
   git add cpp/tests/unit_tests/kernels/CMakeLists.txt
   git add W4A16_INTEGRATION_README.md
   git commit -m "Integrate W4A16 SM90 kernel test using TensorRT-LLM API

   - Add standalone test using CutlassFpAIntBGemmRunner
   - Add GTest unit test for CI integration
   - Use proper TensorRT-LLM API instead of direct launcher calls
   - This should fix the segfault issue from previous standalone approach"
   ```

---

## Troubleshooting

If still getting segfault:
1. Check CUDA version: `nvcc --version` (need 12.x)
2. Check GPU: `nvidia-smi` (should show H800)
3. Enable CUDA error checking: Set `CUDA_LAUNCH_BLOCKING=1`
4. Run with cuda-gdb for more details

---

## References

- Original standalone extraction: `/home/qianxu/trt_llm_w4a16_hopper/`
- Status report: `/home/qianxu/trt_llm_w4a16_hopper/STATUS_REPORT.md`
- TensorRT-LLM repo: [https://github.com/NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- GitHub branch: `w4a16_hopper_extraction`

---

**Author**: Claude Code
**Last Updated**: 2025-11-14
