#!/bin/bash

set -e

echo "=== Building W4A16 SM90 Standalone Test ==="
echo ""
echo "This test uses TensorRT-LLM's CutlassFpAIntBGemmRunner API"
echo "to properly call the W4A16 Hopper kernels."
echo ""

# Create build directory
BUILD_DIR="build_w4a16_test"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "Running CMake..."
# Create a temporary link to our CMakeLists
ln -sf CMakeLists_w4a16_test.txt ../CMakeLists.txt
cmake ..

echo ""
echo "Building with $(nproc) cores..."
make -j$(nproc) w4a16_test

echo ""
echo "✅ Build successful!"
echo ""
echo "Executable: $BUILD_DIR/w4a16_test"
echo ""
echo "To run on H800:"
echo "  cd $BUILD_DIR"
echo "  ./w4a16_test [M] [N] [K]"
echo ""
echo "Example:"
echo "  ./w4a16_test 1024 2048 2048"
