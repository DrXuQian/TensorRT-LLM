# W4A16 SM90 Kernels - H800 Quick Start

## On H800 Machine

### 1. Clone and Setup
```bash
# Clone the repository
git clone https://github.com/DrXuQian/TensorRT-LLM.git
cd TensorRT-LLM

# Checkout W4A16 branch
git checkout w4a16_integration
```

### 2. Compile (Two Options)

#### Option A: Use Pre-filtered W4A16-only Kernels (Recommended)
```bash
# Build with 12 FP16+INT4 kernels (NO FP8/BF16)
./build_w4a16_only.sh
```

**Build time**: ~5-10 minutes with `-j2`
**Output**: `build_w4a16_only/w4a16_only_test`

#### Option B: Generate Fresh Kernels
```bash
# Generate SM90 kernels (48 files, includes FP8/BF16)
export PYTHONPATH=/path/to/TensorRT-LLM/3rdparty/cutlass/python:$PYTHONPATH
cd cpp/tensorrt_llm/kernels/cutlass_kernels/python
python3 generate_kernels.py -a "90" -o /tmp/w4a16_gen

# Copy to project
cd /path/to/TensorRT-LLM
rm -rf generated_kernels
mkdir -p generated_kernels
cp /tmp/w4a16_gen/gemm/90/*.cu generated_kernels/

# Filter to W4A16-only
rm -rf generated_kernels_w4a16_only
mkdir -p generated_kernels_w4a16_only
cd generated_kernels
for f in *.cu; do
    grep -q "__nv_fp8\|__nv_bfloat16" "$f" || cp "$f" ../generated_kernels_w4a16_only/
done

# Build
cd ..
./build_w4a16_only.sh
```

### 3. Run Tests

#### Basic Test
```bash
./build_w4a16_only/w4a16_only_test 512 1024 1024
```

Expected output:
```
=== W4A16 SM90 Minimal Test ===

GPU: NVIDIA H800
Compute Capability: 9.0

Matrix: M=512, N=1024, K=1024, group_size=128

Allocating memory...
Initializing data...

=== Creating CutlassFpAIntBGemmRunner ===
Runner created!
Workspace: X.XX MB
Available configs: XXX
Using first config

=== Running GEMM ===
GEMM call returned
Synchronizing...

✅ Test completed successfully!
```

#### Performance Test (Different Sizes)
```bash
# Small
./build_w4a16_only/w4a16_only_test 128 256 256

# Medium
./build_w4a16_only/w4a16_only_test 1024 2048 2048

# Large
./build_w4a16_only/w4a16_only_test 4096 8192 8192
```

### 4. Verify GPU

Check you're on H100/H800:
```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

Should show:
```
name, compute_cap
NVIDIA H800, 9.0
```
or
```
name, compute_cap
NVIDIA H100, 9.0
```

---

## What's Included

### Kernel Details
- **12 kernel files** in `generated_kernels_w4a16_only/`
- **Pure W4A16**: FP16 activations + INT4 weights
- **NO FP8**: No `__nv_fp8_e4m3` types
- **NO BF16**: No `__nv_bfloat16` types

### Kernel Configurations
| File | CTA Shape | Quantization |
|------|-----------|--------------|
| M128_group11-14 | 128x64x64 variants | PER_COLUMN_SCALE_ONLY |
| M128_group21-25 | 128x128x64, 128x256x64 | FINEGRAINED_SCALE_ONLY |
| M64_group6,11,12 | 64x variants | Mixed |

All use:
- **TMA** (Tensor Memory Accelerator)
- **WGMMA** (Warp Group Matrix Multiply Accumulate)
- **Cluster shapes**: 1x1x1, 2x1x1, 1x2x1, 2x2x1

---

## Troubleshooting

### Build Fails with FP8 Errors
→ You're using the wrong kernels. Use `build_w4a16_only.sh` which uses `generated_kernels_w4a16_only/`

### Runtime Error: "unspecified launch failure"
Check GPU:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
```

If not `9.0`:
- SM 8.0-8.9 (RTX 3090/4090): Need to generate SM80 kernels
- SM 10.0+ (RTX 5070): Falls back to SM80, need those kernels

### Memory Issues During Build
The build script uses `-j2` (2 cores). If still failing:
```bash
# Edit build_w4a16_only.sh, change:
make -j2 w4a16_only_test
# to:
make -j1 w4a16_only_test
```

---

## Next Steps

### Benchmark Performance
Modify `w4a16_minimal_test.cu` to:
- Run multiple iterations
- Measure throughput (TFLOPS)
- Test different group sizes (64, 128, 256)

### Integrate into Application
Use the `CutlassFpAIntBGemmRunner` API:
```cpp
#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"

CutlassFpAIntBGemmRunner<half, cutlass::uint4b_t,
    cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> runner;

runner.gemm(d_A, d_B, d_scales, nullptr, nullptr, 1.0f,
    d_C, M, N, K, group_size, gemm_config, workspace,
    workspace_bytes, stream);
```

### Compare with Full TensorRT-LLM
Build full TensorRT-LLM and compare performance with the extracted kernels.

---

## Files Reference

| File | Purpose |
|------|---------|
| `build_w4a16_only.sh` | Build FP8-free W4A16 kernels |
| `w4a16_minimal_test.cu` | Test program |
| `generated_kernels_w4a16_only/*.cu` | 12 W4A16 kernel files |
| `W4A16_SM90_SUCCESS.md` | Detailed documentation |
| `build_w4a16_only/w4a16_only_test` | Compiled executable |

---

**Ready to run on H800!** 🚀
