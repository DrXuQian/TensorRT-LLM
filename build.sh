#!/bin/bash

# W4A16 Hopper Kernel Build Script

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== W4A16 Hopper (SM90) Kernel Build Script ===${NC}"

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}ERROR: nvcc not found. Please install CUDA toolkit.${NC}"
    exit 1
fi

NVCC_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9\.]*\).*/\1/p')
echo -e "${GREEN}Found CUDA version: $NVCC_VERSION${NC}"

# Find CUTLASS
if [ -z "$CUTLASS_DIR" ]; then
    # Try common locations
    if [ -d "/usr/local/include/cutlass" ]; then
        export CUTLASS_DIR="/usr/local"
    elif [ -d "/usr/include/cutlass" ]; then
        export CUTLASS_DIR="/usr"
    elif [ -d "$HOME/cutlass" ]; then
        export CUTLASS_DIR="$HOME/cutlass"
    else
        echo -e "${YELLOW}WARNING: CUTLASS not found in standard locations.${NC}"
        echo -e "${YELLOW}Please set CUTLASS_DIR environment variable.${NC}"
        echo -e "${YELLOW}Trying to use system includes...${NC}"
        export CUTLASS_DIR="/usr/local"
    fi
fi

echo -e "${GREEN}Using CUTLASS from: $CUTLASS_DIR${NC}"

# Create build directory
BUILD_DIR="build"
if [ -d "$BUILD_DIR" ]; then
    echo -e "${YELLOW}Removing existing build directory...${NC}"
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
echo -e "${GREEN}Configuring with CMake...${NC}"
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUTLASS_DIR="$CUTLASS_DIR" \
    -DCMAKE_CUDA_ARCHITECTURES=90 \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Build
echo -e "${GREEN}Building...${NC}"
make -j$(nproc) VERBOSE=1 2>&1 | tee build.log

if [ $? -eq 0 ]; then
    echo -e "${GREEN}=== Build Successful! ===${NC}"
    echo -e "${GREEN}Library: build/lib/libw4a16_sm90_kernel.so${NC}"
    echo -e "${GREEN}Test executable: build/bin/test_w4a16_sm90${NC}"
else
    echo -e "${RED}=== Build Failed ===${NC}"
    echo -e "${RED}Check build.log for details${NC}"
    exit 1
fi
