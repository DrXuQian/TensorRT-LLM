# TensorRT-LLM FP16-INT4 Kernel Extraction Guide

This guide documents the step-by-step process to extract the FP16-INT4 GEMM kernel from TensorRT-LLM.

## Kernel to Extract
`CutlassFpAIntBGemmRunner<half, cutlass::uint4b_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY>`

## Steps

1. **Copy kernel files from TensorRT-LLM** (unmodified)
   - fpA_intB_gemm.h
   - fpA_intB_gemm_template.h
   - fpA_intB_gemm_template_sm90.h
   - launchers/
   - CUTLASS extensions
   - Common utilities

2. **Create kernel instantiation**
   - Template instantiation for specific types

3. **Add CMakeLists.txt**
   - Link with CUTLASS
   - Set CUDA architectures

4. **Fix compilation issues**
   - Add missing headers
   - Create stub implementations
   - Handle SM90/SM120 compatibility

5. **Add test programs**
   - Simple functionality test
   - Performance benchmark
   - Real weights test

## Target Performance
40+ TFLOPS on RTX 5070 (SM120)