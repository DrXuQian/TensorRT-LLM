// Minimal stub for NvInferRuntime.h
#pragma once

namespace nvinfer1 {
    enum class DataType : int {
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
        kFP4 = 10
    };
}
