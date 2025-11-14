/*
 * W4A16 Hopper Kernel Benchmark
 *
 * 支持配置输入尺寸、量化模式，随机生成数据并真实执行kernel
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cutlass/numeric_types.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <random>
#include <chrono>

// 声明 kernel 函数
extern "C" void w4a16_sm90_gemm_128(
    half const* A,
    cutlass::uint4b_t const* B,
    half const* weight_scales,
    half const* weight_zero_points,
    half const* biases,
    float const alpha,
    half* C,
    int m, int n, int k,
    int const group_size,
    void* gemm_config,
    char* workspace,
    size_t workspace_bytes,
    cudaStream_t stream,
    int* occupancy
);

extern "C" void w4a16_sm90_gemm_64(
    half const* A,
    cutlass::uint4b_t const* B,
    half const* weight_scales,
    half const* weight_zero_points,
    half const* biases,
    float const alpha,
    half* C,
    int m, int n, int k,
    int const group_size,
    void* gemm_config,
    char* workspace,
    size_t workspace_bytes,
    cudaStream_t stream,
    int* occupancy
);

// 错误检查宏
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// 随机初始化 FP16 数据
void random_init_fp16(half* data, size_t n) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (size_t i = 0; i < n; i++) {
        data[i] = __float2half(dis(gen));
    }
}

// 随机初始化 INT4 数据（打包格式）
void random_init_int4(uint8_t* data, size_t n_elements) {
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dis(0, 15); // 4-bit: 0-15

    size_t n_bytes = (n_elements + 1) / 2; // 每个字节存储2个INT4
    for (size_t i = 0; i < n_bytes; i++) {
        uint8_t val1 = dis(gen) & 0xF;
        uint8_t val2 = dis(gen) & 0xF;
        data[i] = (val2 << 4) | val1;
    }
}

// 量化模式
enum QuantMode {
    SCALE_ONLY,      // 仅 scale
    SCALE_AND_ZERO   // scale + zero_point
};

// 配置结构
struct BenchmarkConfig {
    int M, N, K;
    int group_size;
    QuantMode quant_mode;
    bool use_bias;
    int warmup_iters;
    int bench_iters;
    bool use_small_kernel; // true = 64x128, false = 128x128
};

void print_config(const BenchmarkConfig& cfg) {
    printf("=== Benchmark Configuration ===\n");
    printf("Matrix Size: M=%d, N=%d, K=%d\n", cfg.M, cfg.N, cfg.K);
    printf("Group Size: %d\n", cfg.group_size);
    printf("Quantization: %s\n",
           cfg.quant_mode == SCALE_ONLY ? "Scale-Only" : "Scale+Zero");
    printf("Bias: %s\n", cfg.use_bias ? "Yes" : "No");
    printf("Kernel: %s\n",
           cfg.use_small_kernel ? "64x128x128 (Pingpong)" : "128x128x128 (Cooperative)");
    printf("Warmup Iterations: %d\n", cfg.warmup_iters);
    printf("Benchmark Iterations: %d\n", cfg.bench_iters);
    printf("================================\n\n");
}

void run_benchmark(const BenchmarkConfig& cfg) {
    printf("Initializing benchmark...\n");

    // 计算内存大小
    size_t size_A = cfg.M * cfg.K;
    size_t size_B_elements = cfg.N * cfg.K;
    size_t size_B_bytes = (size_B_elements + 1) / 2; // INT4 打包
    size_t size_C = cfg.M * cfg.N;
    size_t num_groups = (cfg.K + cfg.group_size - 1) / cfg.group_size;
    size_t size_scales = cfg.N * num_groups;

    printf("Memory allocation:\n");
    printf("  A (FP16): %.2f MB\n", size_A * sizeof(half) / 1024.0 / 1024.0);
    printf("  B (INT4): %.2f MB\n", size_B_bytes / 1024.0 / 1024.0);
    printf("  C (FP16): %.2f MB\n", size_C * sizeof(half) / 1024.0 / 1024.0);
    printf("  Scales: %.2f MB\n", size_scales * sizeof(half) / 1024.0 / 1024.0);

    // 分配 host 内存
    half* h_A = (half*)malloc(size_A * sizeof(half));
    uint8_t* h_B = (uint8_t*)malloc(size_B_bytes);
    half* h_C = (half*)malloc(size_C * sizeof(half));
    half* h_scales = (half*)malloc(size_scales * sizeof(half));
    half* h_zeros = cfg.quant_mode == SCALE_AND_ZERO ?
                    (half*)malloc(size_scales * sizeof(half)) : nullptr;
    half* h_bias = cfg.use_bias ? (half*)malloc(cfg.N * sizeof(half)) : nullptr;

    // 随机初始化
    printf("Generating random data...\n");
    random_init_fp16(h_A, size_A);
    random_init_int4(h_B, size_B_elements);
    random_init_fp16(h_scales, size_scales);
    if (h_zeros) random_init_fp16(h_zeros, size_scales);
    if (h_bias) random_init_fp16(h_bias, cfg.N);

    // 分配 device 内存
    half *d_A, *d_C, *d_scales, *d_zeros = nullptr, *d_bias = nullptr;
    uint8_t* d_B;
    char* d_workspace;

    size_t workspace_size = 8 * 1024 * 1024; // 8MB workspace

    CUDA_CHECK(cudaMalloc(&d_A, size_A * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, size_B_bytes));
    CUDA_CHECK(cudaMalloc(&d_C, size_C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_scales, size_scales * sizeof(half)));
    if (h_zeros) CUDA_CHECK(cudaMalloc(&d_zeros, size_scales * sizeof(half)));
    if (h_bias) CUDA_CHECK(cudaMalloc(&d_bias, cfg.N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));

    // 拷贝数据到 device
    printf("Copying data to device...\n");
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scales, h_scales, size_scales * sizeof(half), cudaMemcpyHostToDevice));
    if (h_zeros) CUDA_CHECK(cudaMemcpy(d_zeros, h_zeros, size_scales * sizeof(half), cudaMemcpyHostToDevice));
    if (h_bias) CUDA_CHECK(cudaMemcpy(d_bias, h_bias, cfg.N * sizeof(half), cudaMemcpyHostToDevice));

    // 创建 CUDA 流
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // 选择 kernel 函数
    auto kernel_func = cfg.use_small_kernel ? w4a16_sm90_gemm_64 : w4a16_sm90_gemm_128;
    const char* kernel_name = cfg.use_small_kernel ? "w4a16_sm90_gemm_64" : "w4a16_sm90_gemm_128";

    printf("\nRunning kernel: %s\n", kernel_name);

    // Warmup
    printf("Warmup (%d iterations)...\n", cfg.warmup_iters);
    for (int i = 0; i < cfg.warmup_iters; i++) {
        kernel_func(
            d_A,
            reinterpret_cast<cutlass::uint4b_t const*>(d_B),
            d_scales,
            d_zeros,
            d_bias,
            1.0f,
            d_C,
            cfg.M, cfg.N, cfg.K,
            cfg.group_size,
            nullptr, // default config
            d_workspace,
            workspace_size,
            stream,
            nullptr
        );
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Benchmark
    printf("Benchmarking (%d iterations)...\n", cfg.bench_iters);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));

    for (int i = 0; i < cfg.bench_iters; i++) {
        kernel_func(
            d_A,
            reinterpret_cast<cutlass::uint4b_t const*>(d_B),
            d_scales,
            d_zeros,
            d_bias,
            1.0f,
            d_C,
            cfg.M, cfg.N, cfg.K,
            cfg.group_size,
            nullptr,
            d_workspace,
            workspace_size,
            stream,
            nullptr
        );
    }

    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // 计算性能指标
    float avg_time_ms = milliseconds / cfg.bench_iters;

    // GFLOPS 计算: 2*M*N*K operations
    double gflops = (2.0 * cfg.M * cfg.N * cfg.K) / (avg_time_ms / 1000.0) / 1e9;

    // 内存带宽 (简化计算)
    double bytes_read = size_A * sizeof(half) + size_B_bytes + size_scales * sizeof(half);
    double bytes_write = size_C * sizeof(half);
    double bandwidth_gb = (bytes_read + bytes_write) / (avg_time_ms / 1000.0) / 1e9;

    printf("\n=== Results ===\n");
    printf("Average Time: %.3f ms\n", avg_time_ms);
    printf("Throughput: %.2f GFLOPS\n", gflops);
    printf("Memory Bandwidth: %.2f GB/s\n", bandwidth_gb);
    printf("===============\n\n");

    // 拷贝结果回 host 进行验证
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C * sizeof(half), cudaMemcpyDeviceToHost));

    // 简单检查输出不是全零或NaN
    int valid_count = 0;
    for (int i = 0; i < std::min(100, (int)size_C); i++) {
        float val = __half2float(h_C[i]);
        if (!isnan(val) && !isinf(val)) {
            valid_count++;
        }
    }
    printf("Output validation: %d/100 values are valid (not NaN/Inf)\n", valid_count);

    // 打印部分输出
    printf("Sample output values (first 10):\n");
    for (int i = 0; i < std::min(10, (int)size_C); i++) {
        printf("  C[%d] = %.4f\n", i, __half2float(h_C[i]));
    }

    // 清理
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_scales);
    if (h_zeros) free(h_zeros);
    if (h_bias) free(h_bias);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_scales));
    if (d_zeros) CUDA_CHECK(cudaFree(d_zeros));
    if (d_bias) CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_workspace));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

void print_usage(const char* prog_name) {
    printf("Usage: %s [options]\n", prog_name);
    printf("Options:\n");
    printf("  -m M              Matrix dimension M (default: 1024)\n");
    printf("  -n N              Matrix dimension N (default: 4096)\n");
    printf("  -k K              Matrix dimension K (default: 4096)\n");
    printf("  -g GROUP_SIZE     Quantization group size (default: 128)\n");
    printf("  -q QUANT_MODE     Quantization mode: 0=scale_only, 1=scale_and_zero (default: 0)\n");
    printf("  -b                Use bias (default: no bias)\n");
    printf("  -s                Use small kernel (64x128) instead of large (128x128)\n");
    printf("  -w WARMUP         Warmup iterations (default: 5)\n");
    printf("  -i ITERS          Benchmark iterations (default: 10)\n");
    printf("  -h                Show this help\n");
    printf("\nExamples:\n");
    printf("  %s -m 2048 -n 8192 -k 8192 -g 128\n", prog_name);
    printf("  %s -m 512 -n 2048 -k 2048 -q 1 -b\n", prog_name);
}

int main(int argc, char** argv) {
    printf("W4A16 Hopper (SM90) Kernel Benchmark\n");
    printf("=====================================\n\n");

    // 默认配置
    BenchmarkConfig cfg;
    cfg.M = 1024;
    cfg.N = 4096;
    cfg.K = 4096;
    cfg.group_size = 128;
    cfg.quant_mode = SCALE_ONLY;
    cfg.use_bias = false;
    cfg.warmup_iters = 5;
    cfg.bench_iters = 10;
    cfg.use_small_kernel = false;

    // 解析命令行参数
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            cfg.M = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            cfg.N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) {
            cfg.K = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-g") == 0 && i + 1 < argc) {
            cfg.group_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-q") == 0 && i + 1 < argc) {
            cfg.quant_mode = atoi(argv[++i]) == 0 ? SCALE_ONLY : SCALE_AND_ZERO;
        } else if (strcmp(argv[i], "-b") == 0) {
            cfg.use_bias = true;
        } else if (strcmp(argv[i], "-s") == 0) {
            cfg.use_small_kernel = true;
        } else if (strcmp(argv[i], "-w") == 0 && i + 1 < argc) {
            cfg.warmup_iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            cfg.bench_iters = atoi(argv[++i]);
        }
    }

    // 验证 group_size
    if (cfg.K % cfg.group_size != 0) {
        fprintf(stderr, "Error: K (%d) must be divisible by group_size (%d)\n",
                cfg.K, cfg.group_size);
        return 1;
    }

    print_config(cfg);

    // 检查 CUDA 设备
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "No CUDA devices found!\n");
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("CUDA Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Total Memory: %.2f GB\n\n", prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);

    // 运行 benchmark
    try {
        run_benchmark(cfg);
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }

    printf("\nBenchmark completed successfully!\n");

    return 0;
}
