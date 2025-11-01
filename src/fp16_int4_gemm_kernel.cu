// Template instantiation for TensorRT-LLM FP16-INT4 GEMM kernel
#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"
#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm_template.h"

// Explicit template instantiation for FP16 input, INT4 weights with fine-grained quantization
template class tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<
    half,                                                              // ActivationType
    cutlass::uint4b_t,                                                // WeightType
    cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY               // QuantOp
>;