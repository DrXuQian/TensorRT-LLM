/*
 * Minimal CUDA utilities for the standalone fused MoE GEMM build.
 */

#pragma once

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/config.h"
#include <cuda_runtime.h>
#include <cstdio>

TRTLLM_NAMESPACE_BEGIN

namespace common
{
inline void check_cuda_error(cudaError_t status)
{
    if (status != cudaSuccess)
    {
        std::fprintf(stderr, "CUDA error: %s (%d)\n", cudaGetErrorString(status), static_cast<int>(status));
        std::abort();
    }
}
} // namespace common

TRTLLM_NAMESPACE_END
