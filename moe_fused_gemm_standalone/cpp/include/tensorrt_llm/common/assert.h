/*
 * Minimal assert utilities for the standalone fused MoE GEMM build.
 */

#pragma once

#include "tensorrt_llm/common/config.h"
#include <cstdio>
#include <cstdlib>

#define TLLM_CHECK(val)                                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(val))                                                                                                    \
        {                                                                                                              \
            std::fprintf(stderr, "TLLM_CHECK failed: %s (%s:%d)\n", #val, __FILE__, __LINE__);                         \
            std::abort();                                                                                              \
        }                                                                                                              \
    } while (0)

#define TLLM_CHECK_WITH_INFO(val, info, ...)                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(val))                                                                                                    \
        {                                                                                                              \
            std::fprintf(stderr, "TLLM_CHECK failed: " info " (%s:%d)\n", ##__VA_ARGS__, __FILE__, __LINE__);          \
            std::abort();                                                                                              \
        }                                                                                                              \
    } while (0)
