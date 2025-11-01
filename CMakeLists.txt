cmake_minimum_required(VERSION 3.18)
project(trt_llm_flash_attention CUDA CXX)

# Set standards
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# CUDA architectures
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90" CACHE STRING "CUDA architectures")
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Definitions
add_definitions(-DENABLE_FP8=1)
add_definitions(-DENABLE_BF16=1)

# Collect all cubin files
file(GLOB CUBIN_FILES src/cubin/*.cpp)

# Create library
add_library(trt_llm_flash_attention SHARED
    src/fused_multihead_attention_v2.cpp
    src/fmhaRunner.cpp
    src/fmhaPackedMask.cu
    ${CUBIN_FILES}
)

target_link_libraries(trt_llm_flash_attention
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::cublas
    CUDA::cublasLt
)

# Test program
add_executable(test_flash_attention
    test/test_flash_attention.cpp
)

target_link_libraries(test_flash_attention
    trt_llm_flash_attention
    CUDA::cudart
)

# Print configuration
message(STATUS "===== Configuration =====")
message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Number of cubin files: ${CMAKE_MATCH_0}")
message(STATUS "=========================")