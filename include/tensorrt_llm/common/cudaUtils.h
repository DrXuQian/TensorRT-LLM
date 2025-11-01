#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <string>

namespace tensorrt_llm {
namespace common {

// CUDA error checking macro
#define CUDA_CHECK(call)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t error = call;                                                                                     \
        if (error != cudaSuccess)                                                                                     \
        {                                                                                                              \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error)      \
                      << std::endl;                                                                                    \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

inline int getSMVersion() {
    int device_id;
    cudaGetDevice(&device_id);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    return prop.major * 10 + prop.minor;
}

inline int getMultiProcessorCount() {
    int device_id;
    cudaGetDevice(&device_id);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    return prop.multiProcessorCount;
}

inline void sync_check_cuda_error() {
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        exit(1);
    }
}

inline void check_cuda_error(cudaError_t error) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        exit(1);
    }
}

} // namespace common
} // namespace tensorrt_llm