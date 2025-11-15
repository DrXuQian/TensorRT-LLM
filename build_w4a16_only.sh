#!/bin/bash

set -e

echo "=== Building W4A16 (FP16+INT4 only, NO FP8/BF16) ==="
echo ""

BUILD_DIR="build_w4a16_only"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Create CMakeLists
cat > "$BUILD_DIR/CMakeLists.txt" << 'EOF'
cmake_minimum_required(VERSION 3.18)
project(w4a16_only CUDA CXX)

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

# Only use W4A16 (FP16+INT4) kernels - 12 files without FP8/BF16
file(GLOB W4A16_KERNELS "${TRTLLM}/generated_kernels_w4a16_only/*.cu")

message(STATUS "Building with ${CMAKE_MATCH_COUNT} W4A16-only kernels (FP16+INT4)")

add_executable(w4a16_only_test
    ${TRTLLM}/w4a16_minimal_test.cu
    ${W4A16_KERNELS}
    ${TRTLLM}/cpp/tensorrt_llm/common/logger.cpp
    ${TRTLLM}/cpp/tensorrt_llm/common/stringUtils.cpp
    ${TRTLLM}/cpp/tensorrt_llm/common/assert.cpp
    ${TRTLLM}/cpp/tensorrt_llm/common/tllmException.cpp
    ${TRTLLM}/cpp/tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.cpp
)

target_link_libraries(w4a16_only_test CUDA::cudart)

set_target_properties(w4a16_only_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

message(STATUS "W4A16 Kernels: ${W4A16_KERNELS}")
EOF

cd "$BUILD_DIR"
echo "Running CMake..."
cmake .

echo ""
echo "Compiling (using -j2 to avoid memory issues)..."
echo "This will take a few minutes..."
make -j2 w4a16_only_test

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Build successful!"
    echo ""
    echo "Run: ./build_w4a16_only/w4a16_only_test [M] [N] [K]"
    echo "Example: ./build_w4a16_only/w4a16_only_test 512 1024 1024"
else
    echo ""
    echo "❌ Build failed"
    exit 1
fi
