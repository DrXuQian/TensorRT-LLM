#!/bin/bash

set -e

echo "=== 使用生成的 SM90 Kernel 实例化构建 W4A16 ==="
echo ""

BUILD_DIR="build_with_generated"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "Running CMake..."
ln -sf CMakeLists_with_generated.txt ../CMakeLists.txt
cmake ..

echo ""
echo "Building (this may take a while with 48 kernel files)..."
make -j12 w4a16_final

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Build successful!"
    echo ""
    echo "Run: ./build_with_generated/w4a16_final [M] [N] [K]"
    echo "Example: ./build_with_generated/w4a16_final 512 1024 1024"
else
    echo ""
    echo "❌ Build failed"
    exit 1
fi
