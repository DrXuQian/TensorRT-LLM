// Minimal stub for NvInferRuntime.h to avoid TensorRT dependency
#pragma once

#include <cstdint>

namespace nvinfer1 {

// Minimal DataType enum for compilation
enum class DataType : int32_t {
    kFLOAT = 0,
    kHALF = 1,
    kINT8 = 2,
    kINT32 = 3,
    kBOOL = 4,
    kUINT8 = 5,
    kFP8 = 6,
    kBF16 = 7,
    kINT64 = 8,
    kINT4 = 9,
    kFP4 = 10,  // Added for FP4 support
};

}  // namespace nvinfer1