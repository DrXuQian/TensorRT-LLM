#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

// BF16 wrapper utilities
#ifdef ENABLE_BF16

// BF16 is enabled, use native types
using __nv_bfloat16 = __nv_bfloat16;
using __nv_bfloat162 = __nv_bfloat162;

#else

// BF16 not enabled, provide fallback
struct __nv_bfloat16 {
    unsigned short x;
};

struct __nv_bfloat162 {
    __nv_bfloat16 x;
    __nv_bfloat16 y;
};

#endif // ENABLE_BF16