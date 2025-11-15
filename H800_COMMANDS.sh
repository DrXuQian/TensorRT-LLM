#!/bin/bash
# W4A16 SM90 Kernels - H800 Deployment Commands
# Copy and paste these commands on your H800 machine

# ========================================
# Step 1: Clone Repository
# ========================================
git clone https://github.com/DrXuQian/TensorRT-LLM.git
cd TensorRT-LLM
git checkout w4a16_integration

# ========================================
# Step 2: Verify GPU (Optional but Recommended)
# ========================================
echo "Checking GPU..."
nvidia-smi --query-gpu=name,compute_cap --format=csv

# Expected output:
# name, compute_cap
# NVIDIA H800, 9.0

# ========================================
# Step 3: Build W4A16 Kernels (FP8-free)
# ========================================
echo ""
echo "Building W4A16-only kernels..."
./build_w4a16_only.sh

# Build should complete in ~5-10 minutes
# Expected output: "✅ Build successful!"

# ========================================
# Step 4: Run Tests
# ========================================

# Basic test (512x1024x1024)
echo ""
echo "Running basic test..."
./build_w4a16_only/w4a16_only_test 512 1024 1024

# Small test
echo ""
echo "Running small matrix test..."
./build_w4a16_only/w4a16_only_test 128 256 256

# Medium test
echo ""
echo "Running medium matrix test..."
./build_w4a16_only/w4a16_only_test 1024 2048 2048

# Large test (may need more GPU memory)
echo ""
echo "Running large matrix test..."
./build_w4a16_only/w4a16_only_test 4096 8192 8192

echo ""
echo "================================"
echo "All tests completed!"
echo "================================"

# ========================================
# Expected Successful Output:
# ========================================
# === W4A16 SM90 Minimal Test ===
#
# GPU: NVIDIA H800
# Compute Capability: 9.0
#
# Matrix: M=512, N=1024, K=1024, group_size=128
#
# Allocating memory...
# Initializing data...
#
# === Creating CutlassFpAIntBGemmRunner ===
# Runner created!
# Workspace: X.XX MB
# Available configs: XXX
# Using first config
#
# === Running GEMM ===
# GEMM call returned
# Synchronizing...
#
# ✅ Test completed successfully!
