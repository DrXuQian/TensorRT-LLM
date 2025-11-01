#pragma once

#include "tensorrt_llm/common/tllmException.h"
#include <cassert>

namespace tensorrt_llm {
namespace common {

// Define CHECK and DCHECK macros
#define CHECK(cond) TLLM_CHECK(cond)
#define CHECK_WITH_INFO(cond, info) TLLM_CHECK_WITH_INFO(cond, info)

#ifdef NDEBUG
#define DCHECK(cond) ((void)0)
#define DCHECK_WITH_INFO(cond, info) ((void)0)
#else
#define DCHECK(cond) CHECK(cond)
#define DCHECK_WITH_INFO(cond, info) CHECK_WITH_INFO(cond, info)
#endif

} // namespace common
} // namespace tensorrt_llm