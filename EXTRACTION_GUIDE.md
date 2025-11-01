# TensorRT-LLM Flash Attention (FMHA) Extraction Guide

This guide documents the extraction of Flash Attention kernels from TensorRT-LLM.

## What is being extracted

TensorRT-LLM's implementation of Flash Attention through:
- Fused Multi-Head Attention v2 (FMHA v2)
- Optimized kernels for various architectures (SM80, SM89, SM90)
- Support for different data types (FP16, BF16, FP8)

## Key Components

1. **fused_multihead_attention_v2**: Main FMHA implementation
2. **fmhaRunner**: Runtime dispatcher for different kernel variants
3. **fmhaPackedMask**: Mask handling utilities
4. **Precompiled cubins**: Optimized kernels for specific configurations

## Architecture Support

- SM80 (Ampere)
- SM89 (Ada Lovelace)
- SM90 (Hopper)
- SM120 (Blackwell) - via forward compatibility

## Features

- Flash Attention algorithm for memory-efficient attention
- Support for paged KV cache
- ALiBi positional encoding
- Softcapping
- Various head dimensions (48, 64, 80, 104, 128, 192, 256)