#pragma once

#include <string>
#include <cstdarg>

// Define logging macros
#define TLLM_LOG_DEBUG(...) ((void)0)
#define TLLM_LOG_INFO(...) ((void)0)
#define TLLM_LOG_WARNING(...) ((void)0)
#define TLLM_LOG_ERROR(...) ((void)0)

namespace tensorrt_llm {
namespace common {

// Minimal logger implementation
class Logger {
public:
    Logger();
    static Logger* getLogger();

    template<typename... Args>
    void log(const char* format, Args... args) {
        // Empty implementation
    }
};

// fmtstr support
typedef void (*fmtstr_allocator)(void*, char const*, std::size_t);
void fmtstr_(char const* format, fmtstr_allocator alloc, void* target, va_list args);

template<typename... Ts>
inline std::string fmtstr(char const* format, Ts... args) {
    std::string result;
    // Simplified implementation
    return result;
}

} // namespace common
} // namespace tensorrt_llm