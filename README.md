# TensorRT-LLM Flash Attention Extraction

Extraction of Flash Attention (FMHA v2) kernels from TensorRT-LLM.

## What is Flash Attention?

Flash Attention is a memory-efficient attention algorithm that:
- Reduces memory usage from O(N²) to O(N)
- Improves performance through better memory access patterns
- Enables longer sequence lengths

## TensorRT-LLM Implementation

TensorRT-LLM implements Flash Attention through:
- **FMHA v2**: Fused Multi-Head Attention version 2
- **Precompiled kernels**: 80+ optimized cubin files for different configurations
- **Multiple data types**: FP16, BF16, FP8 support
- **Architecture-specific optimizations**: SM80, SM89, SM90

## Extracted Components

### Core Files
- `fused_multihead_attention_v2.cpp/h`: Main FMHA implementation
- `fmhaRunner.cpp/h`: Runtime dispatcher
- `fmhaPackedMask.cu/h`: Mask utilities

### Precompiled Kernels (80 cubin files)
- Different head dimensions: 32, 40, 48, 64, 72, 80, 96, 104, 128, 160, 192, 256
- Data types: FP16, BF16, FP8 (E4M3)
- Features: ALiBi, softcapping, paged KV cache
- Architectures: SM80, SM89, SM90

## Key Features

- **Flash Attention algorithm**: Memory-efficient attention computation
- **Paged KV cache**: Efficient memory management for KV cache
- **ALiBi positional encoding**: Alternative to absolute/relative position embeddings
- **Softcapping**: Prevent attention scores from growing too large
- **TMA (Tensor Memory Accelerator)**: SM90+ hardware acceleration

## Architecture Support

| Architecture | SM Version | Status |
|-------------|------------|--------|
| Ampere      | SM80       | ✅ Supported |
| Ada Lovelace| SM89       | ✅ Supported |
| Hopper      | SM90       | ✅ Supported |
| Blackwell   | SM120      | ✅ Via SM89 compatibility |

## Dependencies

This extraction requires significant TensorRT-LLM infrastructure:
- CUDA Driver API wrappers
- TensorRT runtime interfaces
- Common utilities and assertions
- cubin loader infrastructure

## Note on Complexity

Flash Attention in TensorRT-LLM is deeply integrated with the framework. A full standalone extraction would require:
1. Extracting the cubin loader infrastructure
2. Implementing TMA descriptor handling
3. Creating minimal runtime interfaces
4. Handling all data type conversions

The current extraction preserves the kernel files but would need additional work to run standalone.

## Performance

TensorRT-LLM's Flash Attention achieves:
- Up to 2-4x speedup over standard attention
- Significantly reduced memory usage
- Support for much longer sequences (up to 128K+ tokens)