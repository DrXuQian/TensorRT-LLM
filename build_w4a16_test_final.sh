#!/bin/bash

set -e

echo "=== 构建 W4A16 测试程序 ==="
echo ""

if [ ! -d "build_kernels_lib" ]; then
    echo "❌ 请先运行 ./build_w4a16_lib.sh 构建 kernel 库"
    exit 1
fi

BUILD_DIR="build_w4a16_final"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# 创建测试程序的 CMakeLists
cat > "$BUILD_DIR/CMakeLists.txt" << 'EOF'
cmake_minimum_required(VERSION 3.18)
project(w4a16_test_final CUDA CXX)

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

# 链接预编译的 kernel 对象文件
add_executable(w4a16_test
    ${TRTLLM}/w4a16_minimal_test.cu
)

target_link_libraries(w4a16_test
    ${TRTLLM}/build_kernels_lib/CMakeFiles/w4a16_kernels_obj.dir/generated_kernels/*.cu.o
    ${TRTLLM}/build_kernels_lib/libtrtllm_common.a
    CUDA::cudart
)

set_target_properties(w4a16_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

message(STATUS "Linking with pre-built kernels")
EOF

cd "$BUILD_DIR"
echo "Running CMake..."
cmake .

echo ""
echo "Building test..."
make -j4 w4a16_test

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 构建成功!"
    echo ""
    echo "运行: ./build_w4a16_final/w4a16_test [M] [N] [K]"
    echo "例如: ./build_w4a16_final/w4a16_test 512 1024 1024"
else
    echo ""
    echo "❌ 构建失败"
    exit 1
fi
