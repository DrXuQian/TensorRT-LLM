#!/bin/bash

set -e

echo "=== Building W4A16 with GEMV + CUTLASS ==="
echo ""

# Get absolute path of TensorRT-LLM directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TRTLLM_PATH="$SCRIPT_DIR"

echo "TensorRT-LLM path: $TRTLLM_PATH"
echo ""

BUILD_DIR="build_w4a16_gemv"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Create CMakeLists with dynamic path
cat > "$BUILD_DIR/CMakeLists.txt" << EOF
cmake_minimum_required(VERSION 3.18)
project(w4a16_gemv CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 90)

find_package(CUDAToolkit REQUIRED)
add_definitions(-DCOMPILE_HOPPER_TMA_GEMMS)

set(TRTLLM "$TRTLLM_PATH")
set(CUTLASS "\${TRTLLM}/3rdparty/cutlass")

include_directories(
    \${TRTLLM}/cpp/include
    \${TRTLLM}/cpp
    \${TRTLLM}/cpp/include_stubs
    \${TRTLLM}/cpp/include/tensorrt_llm/cutlass_extensions/include
    \${TRTLLM}/cpp/tensorrt_llm/cutlass_extensions/include
    \${CUTLASS}/include
    \${CUTLASS}/tools/util/include
    \${CUDAToolkit_INCLUDE_DIRS}
)

set(CMAKE_CUDA_FLAGS "\${CMAKE_CUDA_FLAGS} -Xcompiler=-fPIC")
set(CMAKE_CUDA_FLAGS "\${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "\${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=186")
set(CMAKE_CUDA_FLAGS "\${CMAKE_CUDA_FLAGS} -gencode=arch=compute_90a,code=sm_90a")  # H100/H800

# W4A16 CUTLASS kernels (12 files)
file(GLOB W4A16_KERNELS "\${TRTLLM}/generated_kernels_w4a16_only/*.cu")

# GEMV kernels (19 .cu files from weightOnlyBatchedGemv)
file(GLOB GEMV_SOURCES "\${TRTLLM}/cpp/tensorrt_llm/kernels/weightOnlyBatchedGemv/*.cu")

# Common sources
set(COMMON_SOURCES
    \${TRTLLM}/cpp/tensorrt_llm/common/logger.cpp
    \${TRTLLM}/cpp/tensorrt_llm/common/stringUtils.cpp
    \${TRTLLM}/cpp/tensorrt_llm/common/assert.cpp
    \${TRTLLM}/cpp/tensorrt_llm/common/tllmException.cpp
    \${TRTLLM}/cpp/tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.cpp
)

message(STATUS "W4A16 CUTLASS Kernels: \${W4A16_KERNELS}")
message(STATUS "GEMV Sources: \${GEMV_SOURCES}")

# Build w4a16_gemv_test with both CUTLASS and GEMV kernels
add_executable(w4a16_gemv_test
    \${TRTLLM}/w4a16_gemv_test.cu
    \${W4A16_KERNELS}
    \${GEMV_SOURCES}
    \${COMMON_SOURCES}
)

target_link_libraries(w4a16_gemv_test CUDA::cudart)

set_target_properties(w4a16_gemv_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
EOF

cd "$BUILD_DIR"
echo "Running CMake..."
cmake .

echo ""
echo "Compiling (using -j2 to avoid memory issues)..."
echo "This will take a few minutes..."
make -j2 w4a16_gemv_test

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Build successful!"
    echo ""
    echo "Run: ./build_w4a16_gemv/w4a16_gemv_test [M] [N] [K]"
    echo ""
    echo "Examples:"
    echo "  Small batch (GEMV):    ./build_w4a16_gemv/w4a16_gemv_test 16 1024 1024"
    echo "  Medium batch (GEMV):   ./build_w4a16_gemv/w4a16_gemv_test 32 1024 1024"
    echo "  Large batch (CUTLASS): ./build_w4a16_gemv/w4a16_gemv_test 128 1024 1024"
    echo ""
    echo "The program will automatically:"
    echo "  - Use GEMV for M < 64 (small batch)"
    echo "  - Use CUTLASS for M >= 64 (large batch)"
    echo "  - Benchmark multiple batch sizes (1, 4, 16, 32, 64, 128, 256, 512)"
else
    echo ""
    echo "❌ Build failed"
    exit 1
fi
