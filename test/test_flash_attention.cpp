// Simple test for Flash Attention extraction
#include <iostream>
#include <cuda_runtime.h>
#include "tensorrt_llm/kernels/contextFusedMultiHeadAttention/fused_multihead_attention_v2.h"

int main() {
    std::cout << "Testing TensorRT-LLM Flash Attention extraction" << std::endl;

    // Check CUDA device
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    if (device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "SM: " << prop.major << "." << prop.minor << std::endl;

    // Test basic parameters
    const int batch_size = 1;
    const int seq_len = 128;
    const int head_num = 12;
    const int head_size = 64;

    std::cout << "\nFlash Attention Configuration:" << std::endl;
    std::cout << "- Batch size: " << batch_size << std::endl;
    std::cout << "- Sequence length: " << seq_len << std::endl;
    std::cout << "- Number of heads: " << head_num << std::endl;
    std::cout << "- Head size: " << head_size << std::endl;

    // The actual Flash Attention kernel initialization would go here
    // For now, just verify compilation

    std::cout << "\n✅ Flash Attention extraction compiled successfully!" << std::endl;

    return 0;
}