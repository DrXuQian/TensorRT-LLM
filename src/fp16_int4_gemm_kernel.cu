#include "../include/fpA_intB_gemm.h"
#include "../include/fpA_intB_gemm_template.h"

// 实例化需要的具体 kernel
template class tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<
    half, cutlass::uint4b_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY>;