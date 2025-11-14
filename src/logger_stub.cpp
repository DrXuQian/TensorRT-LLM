/*
 * Minimal stub implementations for TensorRT-LLM Logger and Exception
 * This allows the kernel to compile without the full TensorRT-LLM runtime
 */

#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/tllmException.h"
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace tensorrt_llm
{
namespace common
{

// Stub Logger implementation
class LoggerStub : public Logger
{
public:
    void log(Logger::Level level, const char* message, const char* file, int line) override
    {
        // Simple console output for debugging
        // fprintf(stderr, "[%s] %s (%s:%d)\n", getLevelName(level), message, file, line);
    }
};

static LoggerStub gLogger;

Logger& Logger::getLogger()
{
    return gLogger;
}

const char* Logger::getLevelName(Logger::Level level)
{
    switch (level)
    {
    case Level::DEBUG:
        return "DEBUG";
    case Level::INFO:
        return "INFO";
    case Level::WARNING:
        return "WARNING";
    case Level::ERROR:
        return "ERROR";
    default:
        return "UNKNOWN";
    }
}

// Stub fmtstr implementation
std::string fmtstr_(const char* format, char* (*alloc)(void*, size_t), void* ctx, va_list args)
{
    va_list args_copy;
    va_copy(args_copy, args);

    int size = vsnprintf(nullptr, 0, format, args);
    if (size < 0)
    {
        va_end(args_copy);
        return "";
    }

    std::string result(size + 1, '\0');
    vsnprintf(&result[0], size + 1, format, args_copy);
    va_end(args_copy);

    result.resize(size);
    return result;
}

std::string fmtstr(const char* format, ...)
{
    va_list args;
    va_start(args, format);
    std::string result = fmtstr_(format, nullptr, nullptr, args);
    va_end(args);
    return result;
}

// TllmException implementation
TllmException::TllmException(const char* file, size_t line, const char* message)
    : std::runtime_error(message)
{
    // Store file and line for debugging
}

TllmException::~TllmException() noexcept = default;

} // namespace common
} // namespace tensorrt_llm
