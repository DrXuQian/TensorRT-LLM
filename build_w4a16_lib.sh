#!/bin/bash

set -e

echo "=== 分步构建 W4A16 库（避免内存溢出）==="
echo ""

# 设置 PYTHONPATH
export PYTHONPATH=/home/qianxu/TensorRT-LLM/3rdparty/cutlass/python:$PYTHONPATH

# 1. 生成 kernel 实例化
echo "Step 1: 生成 SM90 kernel 实例化..."
cd /home/qianxu/TensorRT-LLM/cpp/tensorrt_llm/kernels/cutlass_kernels/python
python3 generate_kernels.py -a "90" -o /tmp/w4a16_gen

# 2. 复制到项目
echo "Step 2: 复制生成的文件..."
rm -rf /home/qianxu/TensorRT-LLM/generated_kernels
mkdir -p /home/qianxu/TensorRT-LLM/generated_kernels
cp /tmp/w4a16_gen/gemm/90/*.cu /home/qianxu/TensorRT-LLM/generated_kernels/

# 3. 创建库构建目录
echo "Step 3: 构建 kernel 库..."
cd /home/qianxu/TensorRT-LLM
BUILD_DIR="build_kernels_lib"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# 4. 创建 CMakeLists for library
cat > "$BUILD_DIR/CMakeLists.txt" << 'EOF'
cmake_minimum_required(VERSION 3.18)
project(w4a16_kernels CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 90)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)
add_definitions(-DCOMPILE_HOPPER_TMA_GEMMS)

set(TRTLLM "/home/qianxu/TensorRT-LLM")
set(CUTLASS "${TRTLLM}/3rdparty/cutlass")

include_directories(
    ${TRTLLM}/cpp/include
    ${TRTLLM}/cpp
    ${TRTLLM}/cpp/include_stubs
    ${TRTLLM}/cpp/include/tensorrt_llm/cutlass_extensions/include
    ${TRTLLM}/cpp/tensorrt_llm/cutlass_extensions/include
    ${CUTLASS}/include
    ${CUTLASS}/tools/util/include
)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fPIC")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=186")

# 收集生成的 kernel 文件
file(GLOB KERNEL_SOURCES "${TRTLLM}/generated_kernels/*.cu")

# 创建静态库 - 使用 OBJECT 库避免链接问题
add_library(w4a16_kernels_obj OBJECT ${KERNEL_SOURCES})
set_property(TARGET w4a16_kernels_obj PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Common 源文件
add_library(trtllm_common STATIC
    ${TRTLLM}/cpp/tensorrt_llm/common/logger.cpp
    ${TRTLLM}/cpp/tensorrt_llm/common/stringUtils.cpp
    ${TRTLLM}/cpp/tensorrt_llm/common/assert.cpp
    ${TRTLLM}/cpp/tensorrt_llm/common/tllmException.cpp
    ${TRTLLM}/cpp/tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.cpp
)

message(STATUS "Building kernel library with ${CMAKE_CUDA_ARCHITECTURES} kernels")
EOF

cd "$BUILD_DIR"
echo ""
echo "Running CMake..."
cmake .

echo ""
echo "Building library (单核编译避免内存溢出)..."
echo "这会花一些时间，请耐心等待..."
make -j1 w4a16_kernels_obj trtllm_common

echo ""
echo "✅ 库构建完成!"
echo ""
echo "现在可以构建测试程序了"
