cmake_minimum_required(VERSION 3.18)
project(trt_llm_w4a16_hopper CUDA CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Set CUDA architectures - Hopper is SM90
set(CMAKE_CUDA_ARCHITECTURES 90)

# Add compile definitions for Hopper TMA GEMM support
add_definitions(-DCOMPILE_HOPPER_TMA_GEMMS)

# CUTLASS path - using TensorRT-LLM's bundled CUTLASS
if(NOT DEFINED CUTLASS_DIR)
    set(CUTLASS_DIR "/home/qianxu/TensorRT-LLM/3rdparty/cutlass" CACHE PATH "Path to CUTLASS library")
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/tensorrt_llm/cutlass_extensions/include
    ${CUTLASS_DIR}/include
    ${CUTLASS_DIR}/tools/util/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# CUDA compile options
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fPIC")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wall")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=186")  # Suppress CUTLASS warnings

# Separate debug and release flags
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 -lineinfo")

# Build the W4A16 SM90 kernel library
add_library(w4a16_sm90_kernel SHARED
    src/w4a16_sm90_kernel.cu
    src/logger.cpp
    src/stringUtils.cpp
    src/assert.cpp
    src/tllmException.cpp
)

target_link_libraries(w4a16_sm90_kernel
    CUDA::cudart
)

# Set output directory
set_target_properties(w4a16_sm90_kernel PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# Optional: Create a simple test executable
add_executable(test_w4a16_sm90
    src/test_main.cu
)

target_link_libraries(test_w4a16_sm90
    w4a16_sm90_kernel
    CUDA::cudart
)

set_target_properties(test_w4a16_sm90 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Benchmark executable with full functionality
add_executable(benchmark_w4a16_sm90
    src/benchmark_w4a16.cu
)

target_link_libraries(benchmark_w4a16_sm90
    w4a16_sm90_kernel
    CUDA::cudart
)

set_target_properties(benchmark_w4a16_sm90 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Debug executable to check kernel execution
add_executable(debug_w4a16_sm90
    src/debug_kernel.cu
)

target_link_libraries(debug_w4a16_sm90
    w4a16_sm90_kernel
    CUDA::cudart
)

set_target_properties(debug_w4a16_sm90 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Simple test executable
add_executable(simple_test_sm90
    src/simple_test.cu
)

target_link_libraries(simple_test_sm90
    w4a16_sm90_kernel
    CUDA::cudart
)

set_target_properties(simple_test_sm90 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Test with real bias (not nullptr)
add_executable(test_with_bias
    src/test_with_bias.cu
)

target_link_libraries(test_with_bias
    w4a16_sm90_kernel
    CUDA::cudart
)

set_target_properties(test_with_bias PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Print configuration info
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUTLASS Directory: ${CUTLASS_DIR}")
message(STATUS "CUDA Toolkit Include: ${CUDAToolkit_INCLUDE_DIRS}")
