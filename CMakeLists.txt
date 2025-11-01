cmake_minimum_required(VERSION 3.18)
project(trt_llm_fp16_int4_extraction CUDA CXX)

# Set standards
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Set CUTLASS path
set(CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../TensorRT-LLM/3rdparty/cutlass" CACHE PATH "Path to CUTLASS")

if(NOT EXISTS ${CUTLASS_DIR})
    message(FATAL_ERROR "CUTLASS not found at ${CUTLASS_DIR}")
endif()

# CUDA architectures (excluding SM90)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89" CACHE STRING "CUDA architectures")
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cutlass_extensions/include
    ${CUTLASS_DIR}/include
    ${CUTLASS_DIR}/tools/util/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wno-deprecated-declarations")

# Definitions
add_definitions(-DENABLE_FP8=1)
add_definitions(-DENABLE_BF16=1)

# Create library
add_library(trt_llm_fp16_int4_gemm SHARED
    src/fp16_int4_gemm_kernel.cu
    src/missing_implementations.cpp
)

target_link_libraries(trt_llm_fp16_int4_gemm
    CUDA::cudart
    CUDA::cuda_driver
)

# Test programs
add_executable(test_simple
    test/test_simple.cu
)

target_link_libraries(test_simple
    trt_llm_fp16_int4_gemm
    CUDA::cudart
)

# Print configuration
message(STATUS "===== Configuration =====")
message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION}")
message(STATUS "CUTLASS Directory: ${CUTLASS_DIR}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "=========================")