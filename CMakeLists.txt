cmake_minimum_required(VERSION 3.18)
project(extracted_fp16_int4_gemm CUDA CXX)

# 设置标准
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 CUDA
find_package(CUDAToolkit REQUIRED)

# 设置路径
set(CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/cutlass" CACHE PATH "Path to CUTLASS")

if(NOT EXISTS ${CUTLASS_DIR})
    message(FATAL_ERROR "CUTLASS not found at ${CUTLASS_DIR}")
endif()

# CUDA 架构（排除 SM90）
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89" CACHE STRING "CUDA architectures")
endif()

# 包含路径
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cutlass_extensions/include
    ${CUTLASS_DIR}/include
    ${CUTLASS_DIR}/tools/util/include  # for packed_stride.hpp
    ${CUDAToolkit_INCLUDE_DIRS}
)

# 编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wno-deprecated-declarations")

# 定义
add_definitions(-DENABLE_FP8=1)
add_definitions(-DENABLE_BF16=1)

# 创建库
add_library(tensorrt_llm_fp16_int4_gemm SHARED
    src/fp16_int4_gemm_kernel.cu
)

target_link_libraries(tensorrt_llm_fp16_int4_gemm
    CUDA::cudart
    CUDA::cuda_driver
)

# 打印配置
message(STATUS "===== Configuration =====")
message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION}")
message(STATUS "CUTLASS Directory: ${CUTLASS_DIR}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "==========================")