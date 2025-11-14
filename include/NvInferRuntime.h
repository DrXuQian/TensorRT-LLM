/*
 * Minimal NvInferRuntime.h stub for W4A16 kernel extraction
 * This is a simplified version containing only the DataType enum needed by the kernel
 */

#pragma once

namespace nvinfer1
{

enum class DataType : int32_t
{
    kFLOAT = 0,   //!< FP32 format
    kHALF = 1,    //!< FP16 format
    kINT8 = 2,    //!< INT8 format
    kINT32 = 3,   //!< INT32 format
    kBOOL = 4,    //!< BOOL format
    kUINT8 = 5,   //!< UINT8 format
    kFP8 = 6,     //!< FP8 format (E4M3)
    kBF16 = 7,    //!< BF16 format
    kINT64 = 8,   //!< INT64 format
    kINT4 = 9,    //!< INT4 format
    kFP4 = 10     //!< FP4 format
};

} // namespace nvinfer1
