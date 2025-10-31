// 缺失符号的最小实现
#include <stdexcept>
#include <vector>
#include <cstdarg>
#include <string>
#include "tensorrt_llm/common/tllmException.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.h"

namespace tensorrt_llm {
namespace common {

// TllmException 实现
TllmException::TllmException(const char* file, std::size_t line, const char* msg)
    : std::runtime_error(msg)
{
}

TllmException::~TllmException() noexcept = default;

// Logger 实现
Logger::Logger() {}

Logger* Logger::getLogger() {
    static Logger instance;
    return &instance;
}

// fmtstr_ 实现
void fmtstr_(const char* format, fmtstr_allocator alloc, void* target, va_list args) {
    // 空实现 - 仅用于链接
}

} // namespace common

namespace kernels {
namespace cutlass_kernels {

// get_candidate_configs 实现
std::vector<tensorrt_llm::cutlass_extensions::CutlassGemmConfig>
get_candidate_configs(int sm, int split_k_limit,
                     tensorrt_llm::cutlass_extensions::CutlassGemmConfig::CandidateConfigTypeParam config_type)
{
    std::vector<tensorrt_llm::cutlass_extensions::CutlassGemmConfig> configs;

    // 返回一个默认配置
    tensorrt_llm::cutlass_extensions::CutlassGemmConfig config;
    config.tile_config_sm80 = tensorrt_llm::cutlass_extensions::CutlassTileConfig::CtaShape128x128x64_WarpShape128x32x64;
    config.split_k_style = tensorrt_llm::cutlass_extensions::SplitKStyle::NO_SPLIT_K;
    config.split_k_factor = 1;
    config.stages = 3;

    configs.push_back(config);
    return configs;
}

} // namespace cutlass_kernels
} // namespace kernels
} // namespace tensorrt_llm