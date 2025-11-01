#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

namespace tensorrt_llm {
namespace common {

class TllmException : public std::runtime_error {
public:
    TllmException(char const* file, std::size_t line, char const* msg);
    ~TllmException() noexcept;
};

} // namespace common
} // namespace tensorrt_llm

#define TLLM_CHECK(cond) \
    do { \
        if (!(cond)) { \
            throw ::tensorrt_llm::common::TllmException(__FILE__, __LINE__, #cond); \
        } \
    } while (false)

#define TLLM_CHECK_WITH_INFO(cond, info) \
    do { \
        if (!(cond)) { \
            throw ::tensorrt_llm::common::TllmException(__FILE__, __LINE__, info); \
        } \
    } while (false)

#define TLLM_THROW(...) throw ::tensorrt_llm::common::TllmException(__FILE__, __LINE__, __VA_ARGS__)