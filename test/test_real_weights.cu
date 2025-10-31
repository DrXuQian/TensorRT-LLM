// 使用实际权重文件测试
#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"

using namespace tensorrt_llm::kernels::cutlass_kernels;

bool load_binary_file(const std::string& filename, void* data, size_t size) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open " << filename << std::endl;
        return false;
    }
    file.read(static_cast<char*>(data), size);
    return file.good();
}

int main() {
    std::cout << "\n===== 测试提取的 TensorRT-LLM Kernel =====\n" << std::endl;

    // 检查 GPU
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int sm = prop.major * 10 + prop.minor;
    std::cout << "GPU: " << prop.name << " (SM" << sm << ")" << std::endl;

    // 矩阵维度（匹配权重文件）
    const int m = 1;
    const int n = 11008;
    const int k = 2048;
    const int group_size = 128;

    try {
        // 创建 kernel runner
        CutlassFpAIntBGemmRunner<half, cutlass::uint4b_t,
            cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> runner;
        std::cout << "✅ Kernel runner 创建成功\n" << std::endl;

        // 分配内存
        size_t weight_bytes = (k * n / 2);
        size_t scale_bytes = (k / group_size) * n * sizeof(half);

        half *d_input, *d_output, *d_scales;
        uint8_t *d_weights;

        cudaMalloc(&d_input, m * k * sizeof(half));
        cudaMalloc(&d_weights, weight_bytes);
        cudaMalloc(&d_scales, scale_bytes);
        cudaMalloc(&d_output, m * n * sizeof(half));

        // 尝试加载权重文件
        std::vector<uint8_t> h_weights(weight_bytes);
        std::vector<half> h_scales(scale_bytes / sizeof(half));

        std::string weight_file = "../up_proj_qweight.bin";
        std::string scale_file = "../up_proj_scales.bin";

        bool weights_loaded = false;
        if (load_binary_file(weight_file, h_weights.data(), weight_bytes) &&
            load_binary_file(scale_file, h_scales.data(), scale_bytes)) {
            cudaMemcpy(d_weights, h_weights.data(), weight_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_scales, h_scales.data(), scale_bytes, cudaMemcpyHostToDevice);
            weights_loaded = true;
            std::cout << "✅ 加载实际权重文件成功" << std::endl;
        } else {
            std::cout << "⚠️ 权重文件未找到，使用随机数据" << std::endl;
            cudaMemset(d_weights, 0x88, weight_bytes);
            cudaMemset(d_scales, 0, scale_bytes);
        }

        // 初始化输入
        cudaMemset(d_input, 0, m * k * sizeof(half));
        cudaMemset(d_output, 0, m * n * sizeof(half));

        // 获取 workspace
        size_t workspace_size = runner.getWorkspaceSize(m, n, k);
        void* workspace = nullptr;
        if (workspace_size > 0) {
            cudaMalloc(&workspace, workspace_size);
            std::cout << "Workspace 大小: " << workspace_size << " bytes" << std::endl;
        }

        // 配置
        tensorrt_llm::cutlass_extensions::CutlassGemmConfig config;
        config.tile_config_sm80 = tensorrt_llm::cutlass_extensions::CutlassTileConfig::CtaShape128x128x64_WarpShape128x32x64;
        config.split_k_style = tensorrt_llm::cutlass_extensions::SplitKStyle::NO_SPLIT_K;
        config.split_k_factor = 1;
        config.stages = 3;

        std::cout << "\n执行 kernel..." << std::endl;

        // 运行 kernel
        runner.gemm(d_input, d_weights, d_scales, nullptr, nullptr,
                   1.0f, d_output, m, n, k, group_size, config,
                   (char*)workspace, workspace_size, 0);

        cudaError_t err = cudaDeviceSynchronize();
        if (err == cudaSuccess) {
            std::cout << "\n✅✅✅ TensorRT-LLM FP16-INT4 kernel 执行成功！" << std::endl;

            if (weights_loaded) {
                // 检查输出
                std::vector<half> h_output(m * n);
                cudaMemcpy(h_output.data(), d_output, m * n * sizeof(half), cudaMemcpyDeviceToHost);

                int non_zero = 0;
                for (int i = 0; i < n; ++i) {
                    if (__half2float(h_output[i]) != 0) non_zero++;
                }
                std::cout << "非零输出: " << non_zero << "/" << n << std::endl;
            }

            std::cout << "\n🎉 这是原始的 TensorRT-LLM kernel 在运行！" << std::endl;
        } else {
            std::cout << "❌ CUDA 错误: " << cudaGetErrorString(err) << std::endl;
        }

        // 清理
        cudaFree(d_input);
        cudaFree(d_weights);
        cudaFree(d_scales);
        cudaFree(d_output);
        if (workspace) cudaFree(workspace);

    } catch (const std::exception& e) {
        std::cerr << "异常: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}