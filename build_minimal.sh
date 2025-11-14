#!/bin/bash

set -e

echo "=== 最小化 W4A16 SM90 测试 ==="
echo "只编译一个文件，包含模板实例化"
echo ""

BUILD_DIR="build_minimal"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "Running CMake..."
ln -sf CMakeLists_minimal.txt ../CMakeLists.txt
cmake ..

echo ""
echo "Building..."
make -j12 w4a16_minimal

echo ""
echo "✅ Build successful!"
echo ""
echo "Run: ./build_minimal/w4a16_minimal [M] [N] [K]"
