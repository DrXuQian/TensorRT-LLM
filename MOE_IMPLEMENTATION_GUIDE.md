# TensorRT-LLM MoE (Mixture of Experts) 实现详解

## 目录
1. [概述](#概述)
2. [MoE 架构说明](#moe-架构说明)
3. [核心实现组件](#核心实现组件)
4. [Group GEMM 计算策略](#group-gemm-计算策略)
5. [Prologue 和 Epilogue Fusion](#prologue-和-epilogue-fusion)
6. [性能优化技术](#性能优化技术)
7. [实现细节](#实现细节)

---

## 概述

TensorRT-LLM 中的 MoE (Mixture of Experts) 实现是一个高度优化的 GPU kernel，用于加速大规模语言模型中的专家混合计算。该实现基于 CUTLASS 库，支持多种量化方式、融合操作和并行策略。

### 主要特点
- **高性能 Group GEMM**: 支持将多个专家的 GEMM 操作合并执行
- **多种量化支持**: INT4/INT8/FP8/FP4 权重量化，FP8 激活量化
- **算子融合**: Prologue/Epilogue 融合，减少内存访问
- **灵活的并行策略**: 支持 Tensor Parallelism (TP) 和 Expert Parallelism (EP)
- **动态路由**: 支持 Top-K 专家选择和负载均衡

---

## MoE 架构说明

### 基本流程

```
输入 (Hidden States)
    ↓
路由器 (Router/Gate)
    ↓
Top-K 专家选择
    ↓
Token 重排序 (Permutation)
    ↓
专家计算 (Expert Computation)
    ├── FC1: Hidden → Inter (包含激活函数)
    └── FC2: Inter → Hidden
    ↓
加权求和 (Weighted Sum)
    ↓
输出 (Final Output)
```

### 关键文件结构

```
cpp/tensorrt_llm/
├── kernels/cutlass_kernels/
│   ├── include/
│   │   ├── moe_kernels.h              # MoE 主接口
│   │   ├── moe_gemm_kernels.h         # GEMM kernel 定义
│   │   └── moe_util_kernels.h         # 辅助 kernel
│   └── moe_gemm/
│       ├── moe_gemm_template_dispatch.h
│       └── launchers/
│           └── moe_gemm_tma_ws_launcher.h  # TMA 优化启动器
├── cutlass_extensions/
│   └── include/cutlass_extensions/
│       └── gemm/kernel/
│           ├── moe_cutlass_kernel.h    # CUTLASS kernel 实现
│           └── moe_problem_visitor.h   # 问题访问器
└── thop/
    ├── moeOp.cpp                       # PyTorch 操作接口
    └── moeCommOp.cpp                   # 通信操作
```

---

## 核心实现组件

### 1. CutlassMoeFCRunner

主要的运行器类，负责管理整个 MoE 计算流程：

```cpp
template <typename T,           // 激活类型
          typename WeightType,   // 权重类型
          typename OutputType,   // 输出类型
          typename InputType,    // 输入类型
          typename BackBoneType> // 骨干网络类型
class CutlassMoeFCRunner : public CutlassMoeFCRunnerInterface
{
    // 核心方法
    void runMoe(...);  // 主入口
    void gemm1(...);   // FC1 计算
    void gemm2(...);   // FC2 计算
};
```

### 2. GroupedGemmInput

定义了 Group GEMM 的输入参数：

```cpp
template <typename AType, typename BType, typename BScaleType, typename OType>
struct GroupedGemmInput
{
    AType const* A;                              // 激活矩阵
    BType const* B;                              // 权重矩阵
    BScaleType const* scales;                    // 量化 scale
    BScaleType const* zeros;                     // 量化 zero point
    OType const* biases;                         // 偏置
    OType* C;                                    // 输出

    int64_t const* total_tokens_including_expert; // 每个专家的 token 数
    ActivationType activation_type;              // 激活函数类型
    int64_t num_rows;                            // 总行数
    int64_t n, k;                                // 矩阵维度
    int num_experts;                             // 专家数量
};
```

### 3. TmaWarpSpecializedGroupedGemmInput

Hopper 架构（SM90+）专用的优化输入结构，支持 TMA (Tensor Memory Accelerator)：

```cpp
struct TmaWarpSpecializedGroupedGemmInput
{
    // 融合的 finalize epilogue
    struct FusedFinalizeEpilogue
    {
        void* ptr_final_output;
        void const** ptr_bias;
        float const** ptr_router_scales;
        int const** ptr_source_token_index;
        bool use_reduction;
    };

    enum class EpilogueFusion {
        NONE,
        ACTIVATION,         // 融合激活函数
        GATED_ACTIVATION,   // 融合门控激活（如 SwiGLU）
        FINALIZE           // 融合最终输出
    };
};
```

---

## Group GEMM 计算策略

### 1. 专家分组执行

MoE 通过将多个专家的 GEMM 操作组合成一个 Group GEMM 来提高效率：

```cpp
// 专家问题访问器
template <typename ThreadblockShape>
struct BaseMoeProblemVisitor
{
    // 计算每个专家的问题大小
    cutlass::gemm::GemmCoord problem_size(int idx) const
    {
        const int64_t prev_problem_row = idx == 0 ? 0 :
            params.last_row_for_problem[idx - 1];
        const int64_t current_problem_row =
            params.last_row_for_problem[idx];
        const int64_t gemm_m = current_problem_row - prev_problem_row;

        return GemmCoord(gemm_m, params.gemm_n, params.gemm_k);
    }
};
```

### 2. Token 批处理

将路由到同一专家的 token 批量处理：

```cpp
// Token 重排序
// 1. 根据专家分配对 token 进行排序
// 2. 计算每个专家的起始偏移
// 3. 批量执行 GEMM

for (int expert = 0; expert < num_experts; expert++) {
    int start_idx = expert_first_token_offset[expert];
    int end_idx = expert_first_token_offset[expert + 1];
    int num_tokens = end_idx - start_idx;

    if (num_tokens > 0) {
        // 执行该专家的 GEMM
        gemm_for_expert(expert, start_idx, num_tokens);
    }
}
```

### 3. 动态调度模式

支持三种调度模式：

```cpp
enum class GroupScheduleMode {
    HOST,       // 主机端调度
    DEVICE,     // 设备端动态调度
    PERSISTENT  // 持久化 kernel
};
```

---

## Prologue 和 Epilogue Fusion

### 1. Prologue Fusion (输入处理)

**融合内容**：
- **量化反量化**: FP8/INT4/INT8 输入的反量化
- **Prequant Scale**: AWQ/GPTQ 的预量化缩放
- **Token 聚集**: 根据路由结果聚集 token

```cpp
// Prologue 中的反量化
if (use_fp8) {
    // FP8 反量化融合在 GEMM 的 prologue 中
    dequantized_input = input * dequant_scale;
}

if (use_int4_groupwise) {
    // INT4 group-wise 反量化
    for (int g = 0; g < num_groups; g++) {
        dequantized = (packed_weight - zero[g]) * scale[g];
    }
}
```

### 2. Epilogue Fusion (输出处理)

**融合内容**：

#### a. 激活函数融合 (ACTIVATION)
```cpp
enum class ActivationType {
    Relu,
    Gelu,
    Silu,      // SwiGLU 的一部分
    Swiglu,    // 完整的 SwiGLU
    Geglu,     // Gated GELU
    Identity
};

// 在 epilogue 中融合激活
if (activation_type == ActivationType::Swiglu) {
    // SwiGLU: x * sigmoid(x * beta)
    output = x * sigmoid(x * beta);
}
```

#### b. 门控激活融合 (GATED_ACTIVATION)
对于门控激活（如 SwiGLU），FC1 实际执行两个 GEMM：

```cpp
// FC1 with gated activation
// 输入: [num_tokens, hidden_size]
// 输出1: [num_tokens, inter_size] -> 用于激活
// 输出2: [num_tokens, inter_size] -> 用于门控

intermediate1 = input @ W1;  // 激活路径
intermediate2 = input @ W2;  // 门控路径

// 融合的门控激活
output = activation(intermediate1) * intermediate2;
```

#### c. Finalize 融合 (FINALIZE)
将最终的专家输出聚合融合到 epilogue 中：

```cpp
// Finalize epilogue (FC2 输出处理)
struct FusedFinalizeEpilogue {
    // 1. 应用路由权重
    weighted_output = expert_output * router_scale;

    // 2. Scatter 到原始 token 顺序
    for (int i = 0; i < num_tokens; i++) {
        int orig_idx = permuted_to_unpermuted[i];
        final_output[orig_idx] += weighted_output[i];
    }

    // 3. 可选的偏置添加（仅在 rank 0）
    if (tp_rank == 0 && bias != nullptr) {
        final_output += bias;
    }
};
```

### 3. 融合优势

- **减少内存访问**: 避免中间结果的读写
- **提高缓存利用率**: 数据在寄存器/共享内存中处理
- **减少 kernel 启动开销**: 多个操作在一个 kernel 中完成

---

## 性能优化技术

### 1. TMA (Tensor Memory Accelerator) 优化

Hopper 架构的专用优化：

```cpp
// TMA Warp Specialized 模式
if (moe_gemm_runner_.supportsTmaWarpSpecialized()) {
    // 使用 TMA 进行高效的全局内存访问
    // 支持异步拷贝和 warp 专门化
    use_tma_warp_specialized = true;
}
```

### 2. 量化优化

支持多种量化方案：

```cpp
struct QuantParams {
    // INT 量化
    struct {
        void const* fc1_weight_scales;
        void const* fc2_weight_scales;
    } wo;

    // FP8 量化
    struct {
        float const* dequant_fc1;
        float const* quant_fc2;
        float const* dequant_fc2;
    } fp8;

    // Group-wise 量化
    struct {
        int group_size;
        void const* weight_scales;
        void const* weight_zeros;
    } groupwise;
};
```

### 3. 并行策略

#### Tensor Parallelism (TP)
```cpp
// 权重矩阵在 TP 维度上切分
// FC1: ColumnLinear (切分输出维度)
// FC2: RowLinear (切分输入维度)
inter_size_per_tp = inter_size / tp_size;
```

#### Expert Parallelism (EP)
```cpp
// 专家在节点间分布
experts_per_node = total_experts / ep_size;
start_expert = ep_rank * experts_per_node;
end_expert = (ep_rank + 1) * experts_per_node;
```

### 4. 负载均衡

```cpp
// 最小延迟模式 - 只激活需要的专家
struct MoeMinLatencyParams {
    int* num_active_experts_per_node;  // 活跃专家数
    float* experts_to_token_score;     // 专家分数
    int* active_expert_global_ids;     // 活跃专家 ID
};
```

---

## 实现细节

### 1. 工作空间管理

```cpp
// 计算所需的工作空间
size_t getWorkspaceSize(
    int64_t num_rows,
    int64_t hidden_size,
    int64_t inter_size,
    int num_experts,
    int experts_per_token,
    ActivationType activation_type,
    MOEParallelismConfig parallelism_config)
{
    // 包含：
    // - Token 排序缓冲区
    // - 中间结果缓冲区
    // - GEMM 工作空间
    // - 通信缓冲区（如果需要）
}
```

### 2. 主要执行流程

```cpp
void runMoe(...) {
    // 1. Token 路由和排序
    topk_softmax_and_permute(router_logits, topk_weights, topk_indices);

    // 2. FC1 计算 (Hidden -> Inter)
    gemm1(permuted_input, fc1_weights, fc1_output, ...);

    // 3. 激活函数（可能融合在 FC1 epilogue）
    if (is_gated_activation) {
        gated_activation(fc1_output, gate_output, activated_output);
    }

    // 4. FC2 计算 (Inter -> Hidden)
    gemm2(activated_output, fc2_weights, fc2_output, ...);

    // 5. 最终聚合（可能融合在 FC2 epilogue）
    finalize_output(fc2_output, router_weights, final_output);
}
```

### 3. CUTLASS Kernel 配置

```cpp
// 配置选择
struct CutlassGemmConfig {
    // Tile 配置
    CutlassTileConfigSM90 tile_config_sm90;

    // 调度策略
    MainloopScheduleType mainloop_schedule;  // AUTO/COOPERATIVE/PINGPONG
    EpilogueScheduleType epilogue_schedule;  // AUTO/NO_SMEM/TMA

    // Cluster shape
    ClusterShape cluster_shape;

    // 是否启用 CUDA kernel（用于小批量）
    bool enableCudaKernel;
};
```

---

## 总结

TensorRT-LLM 的 MoE 实现通过以下关键技术实现了高性能：

1. **Group GEMM**: 将多个专家的计算合并，提高 GPU 利用率
2. **算子融合**: Prologue/Epilogue 融合减少内存访问
3. **多种量化**: 支持 INT4/INT8/FP8/FP4 等多种量化方案
4. **硬件优化**: 利用 Hopper TMA 等新硬件特性
5. **灵活并行**: 支持 TP/EP 多种并行策略
6. **动态调度**: 根据负载动态调整执行策略

这些优化使得 MoE 能够在保持模型容量的同时，实现接近稠密模型的推理速度。

---

**相关文件快速索引**：
- 主接口: `cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h`
- GEMM 实现: `cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_gemm_kernels.h`
- CUTLASS kernel: `cpp/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/gemm/kernel/moe_cutlass_kernel.h`
- PyTorch 接口: `cpp/tensorrt_llm/thop/moeOp.cpp`