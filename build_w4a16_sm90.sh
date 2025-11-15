#!/bin/bash

set -e

echo "=== 构建 W4A16 SM90 (只包含 FP16 kernels) ==="
echo ""

BUILD_DIR="build_with_generated"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# 创建 CMakeLists
cat > "$BUILD_DIR/CMakeLists.txt" << 'EOF'
cmake_minimum_required(VERSION 3.18)
project(w4a16_sm90 CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 90)

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

# 只使用 FP16 kernels (19个文件)
file(GLOB FP16_KERNELS "${TRTLLM}/generated_kernels_fp16/*.cu")

message(STATUS "Building with ${CMAKE_MATCH_COUNT} FP16 kernels")

add_executable(w4a16_sm90_test
    ${TRTLLM}/w4a16_minimal_test.cu
    ${FP16_KERNELS}
    ${TRTLLM}/cpp/tensorrt_llm/common/logger.cpp
    ${TRTLLM}/cpp/tensorrt_llm/common/stringUtils.cpp
    ${TRTLLM}/cpp/tensorrt_llm/common/assert.cpp
    ${TRTLLM}/cpp/tensorrt_llm/common/tllmException.cpp
    ${TRTLLM}/cpp/tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.cpp
)

target_link_libraries(w4a16_sm90_test CUDA::cudart)

set_target_properties(w4a16_sm90_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

message(STATUS "Kernels: ${FP16_KERNELS}")
EOF

cd "$BUILD_DIR"
echo "Running CMake..."
cmake .

echo ""
echo "编译 (使用 -j2 限制并行数避免内存溢出)..."
echo "这需要几分钟..."
make -j2 w4a16_sm90_test

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 构建成功!"
    echo ""
    echo "运行: ./build_with_generated/w4a16_sm90_test [M] [N] [K]"
    echo "例如: ./build_with_generated/w4a16_sm90_test 512 1024 1024"
else
    echo ""
    echo "❌ 构建失败"
    exit 1
fi
