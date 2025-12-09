# MoE FC2 Epilogue 写出操作（Scatter）详解

## 概述

FC2 epilogue 可以融合 finalize 操作，包括将专家输出 **scatter（分散写回）** 到原始 token 顺序，并进行加权求和。这是 MoE 计算的最后一步。

## 1. FC2 计算流程

### 完整的 MoE 数据流
```
原始 Token → Gather → FC1 → 激活 → FC2 → Scatter → 最终输出
         ↓聚集            ↓专家计算         ↑分散写回
    [按专家分组]      [批量GEMM]       [恢复原始顺序]
```

## 2. FC2 Epilogue 的 Finalize 融合

### 2.1 传统方式（无融合）

```cpp
// Step 1: FC2 GEMM
// 输入: fc1_output [expanded_num_tokens, inter_size]
// 输出: fc2_output [expanded_num_tokens, hidden_size]
gemm(fc1_output, fc2_weights, fc2_output);

// Step 2: 应用路由权重（单独的 kernel）
apply_router_weights_kernel(fc2_output, router_weights, weighted_output);

// Step 3: Scatter 到原始位置（又一个 kernel）
scatter_kernel(weighted_output, final_output, unpermuted_indices);

// Step 4: 累加多个专家的输出（如果 k > 1）
accumulate_kernel(final_output);
```

### 2.2 融合方式（Finalize Epilogue）

```cpp
// 所有操作融合在 FC2 的 epilogue 中
struct FusedFinalizeEpilogue {
    // 在一个 kernel 中完成：
    // 1. FC2 GEMM 计算
    // 2. 应用路由权重
    // 3. Scatter 到原始位置
    // 4. 累加多专家输出
};
```

## 3. Scatter 操作详细实现

### 3.1 基本 Scatter（写回原始顺序）

```cpp
// 核心 scatter 操作
__global__ void scatter_add_kernel(
    float* final_output,        // [num_tokens, hidden_size] - 最终输出
    const float* expert_output,  // [expanded_num_tokens, hidden_size] - 专家输出
    const float* router_weights, // [expanded_num_tokens] - 路由权重
    const int* permuted_row_to_unpermuted_row,  // 新位置→原始位置映射
    int expanded_num_tokens,
    int hidden_size)
{
    int permuted_idx = blockIdx.x;  // 重排序后的 token 索引
    int feat_idx = threadIdx.x;     // 特征维度索引

    if (permuted_idx < expanded_num_tokens && feat_idx < hidden_size) {
        // 获取原始 token 位置
        int original_token_idx = permuted_row_to_unpermuted_row[permuted_idx];

        // 读取专家输出和路由权重
        float expert_val = expert_output[permuted_idx * hidden_size + feat_idx];
        float weight = router_weights[permuted_idx];

        // 加权后写回原始位置（原子加法，因为可能有多个专家）
        atomicAdd(&final_output[original_token_idx * hidden_size + feat_idx],
                  expert_val * weight);
    }
}
```

### 3.2 融合的 FC2 Epilogue 实现

```cpp
// 在 CUTLASS epilogue 中融合 scatter
template <typename OutputOp>
class MoeFinalizeEpilogue {
public:
    // FC2 GEMM 的 epilogue 阶段
    CUTLASS_DEVICE void operator()(
        FragmentC const& gemm_output,      // GEMM 计算结果
        FragmentC const& bias,              // 偏置（可选）
        FragmentC& final_output)            // 最终输出
    {
        // 1. 获取当前线程处理的 token
        int permuted_token_idx = get_permuted_token_idx();

        // 2. 查找原始 token 位置
        int original_idx = permuted_row_to_unpermuted_row[permuted_token_idx];

        // 3. 获取路由权重
        float router_weight = router_weights[permuted_token_idx];

        // 4. 应用权重和偏置
        for (int i = 0; i < kElementsPerAccess; ++i) {
            // 应用路由权重
            gemm_output[i] *= router_weight;

            // 添加偏置（只在 TP rank 0）
            if (tp_rank == 0 && bias != nullptr) {
                gemm_output[i] += bias[i];
            }
        }

        // 5. Scatter 写回原始位置
        // 注意：这里使用原子操作，因为多个专家可能写入同一位置
        int output_offset = original_idx * hidden_size + thread_feature_idx;
        for (int i = 0; i < kElementsPerAccess; ++i) {
            atomicAdd(&final_output_ptr[output_offset + i], gemm_output[i]);
        }
    }
};
```

## 4. TensorRT-LLM 中的实现

### 4.1 TmaWarpSpecializedGroupedGemmInput 结构

```cpp
struct TmaWarpSpecializedGroupedGemmInput {
    // Finalize epilogue 参数
    struct FusedFinalizeEpilogue {
        // 最终输出位置（原始 token 顺序）
        void* ptr_final_output;

        // 路由器的缩放权重
        float const** ptr_router_scales;

        // Token 索引映射（重排序→原始）
        int const** ptr_source_token_index;

        // 输出维度
        int num_rows_in_final_output;

        // 是否使用 reduction（多专家累加）
        bool use_reduction = true;
    };

    // Epilogue 融合类型
    enum class EpilogueFusion {
        NONE,             // 无融合
        ACTIVATION,       // 融合激活函数
        GATED_ACTIVATION, // 融合门控激活
        FINALIZE         // 融合 finalize（包括 scatter）
    };
};
```

### 4.2 具体的融合实现

```cpp
// 在 moe_gemm_kernels.cu 中
template <typename T>
static void gemm2(
    // ... 参数 ...
    )
{
    // 设置 epilogue 融合
    if (use_finalize_fusion && sm >= 90) {
        hopper_inputs.fusion =
            TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::FINALIZE;

        // 设置 finalize 参数
        hopper_inputs.fused_finalize_epilogue = {
            .ptr_final_output = final_output,
            .ptr_router_scales = router_scales,
            .ptr_source_token_index = unpermuted_row_to_permuted_row,
            .num_rows_in_final_output = num_tokens,
            .use_reduction = (experts_per_token > 1)
        };
    }

    // 执行融合的 GEMM
    moe_gemm_runner.moeGemm(inputs, hopper_inputs);
}
```

## 5. 多专家情况（Top-K, K>1）

当每个 token 选择多个专家时，scatter 操作需要累加：

```cpp
// Token 0 可能被发送给专家 2 和专家 5
// 需要累加两个专家的输出

__global__ void scatter_multi_expert_kernel(
    float* final_output,
    const float* expert_outputs,
    const float* router_weights,
    const int* token_expert_pairs,  // [(token_id, expert_id), ...]
    int num_pairs,
    int hidden_size)
{
    int pair_idx = blockIdx.x;
    int feat_idx = threadIdx.x;

    if (pair_idx < num_pairs && feat_idx < hidden_size) {
        int token_id = token_expert_pairs[pair_idx * 2];
        int expert_id = token_expert_pairs[pair_idx * 2 + 1];

        // 获取该 token-专家对的输出和权重
        float output_val = expert_outputs[pair_idx * hidden_size + feat_idx];
        float weight = router_weights[pair_idx];

        // 累加到最终输出（原子操作）
        atomicAdd(&final_output[token_id * hidden_size + feat_idx],
                  output_val * weight);
    }
}
```

## 6. 性能优化

### 6.1 原子操作优化

```cpp
// 使用 warp-level 原子操作减少冲突
__device__ void warp_atomic_add(float* addr, float val) {
    // 使用 warp shuffle 先在 warp 内部累加
    float warp_sum = warp_reduce_sum(val);

    // 只有 lane 0 执行原子操作
    if (threadIdx.x % 32 == 0) {
        atomicAdd(addr, warp_sum);
    }
}
```

### 6.2 内存访问优化

```cpp
// 使用向量化加载/存储
float4* vec_output = reinterpret_cast<float4*>(final_output);
float4 vec_data = make_float4(val0, val1, val2, val3);

// 向量化原子操作（如果硬件支持）
atomicAdd4(vec_output + offset, vec_data);
```

## 7. 完整的融合流程

```cpp
// FC2 with Finalize Epilogue 完整流程
__device__ void fc2_with_finalize_epilogue(
    // GEMM 输入
    const float* inter_output,      // FC1 输出
    const float* fc2_weights,       // FC2 权重
    const float* fc2_bias,          // FC2 偏置

    // Finalize 参数
    const float* router_weights,    // 路由权重
    const int* scatter_indices,     // scatter 索引

    // 输出
    float* final_output,            // 最终输出（原始顺序）

    // 维度
    int M, int N, int K)
{
    // 1. GEMM 主循环计算
    float accumulator[TILE_SIZE];
    gemm_mainloop(inter_output, fc2_weights, accumulator, M, N, K);

    // 2. Epilogue：融合所有后处理
    #pragma unroll
    for (int i = 0; i < TILE_SIZE; i++) {
        // 2.1 添加偏置
        if (bias != nullptr) {
            accumulator[i] += fc2_bias[i];
        }

        // 2.2 应用路由权重
        float weighted = accumulator[i] * router_weights[token_idx];

        // 2.3 Scatter 到原始位置
        int original_idx = scatter_indices[token_idx];
        atomicAdd(&final_output[original_idx * N + i], weighted);
    }
}
```

## 8. 优势总结

### 融合的优势
1. **减少 kernel 启动开销**：4个操作合并为1个
2. **减少内存带宽**：中间结果保留在寄存器
3. **更好的缓存利用**：数据复用性高
4. **减少同步开销**：不需要多个 kernel 之间的同步

### 性能提升
- **内存带宽节省**：约 2-3x（避免中间结果读写）
- **延迟降低**：约 30-50%（取决于问题规模）
- **吞吐量提升**：特别是在小批量场景下显著

## 9. 配置和使用

```cpp
// 检查是否支持 finalize 融合
bool supports_finalize_fusion =
    (sm >= 90) &&                    // Hopper 或更新
    moe_gemm_runner.supportsTmaWarpSpecialized() &&
    !use_int4_groupwise;              // 某些量化模式不支持

// 启用融合
if (supports_finalize_fusion) {
    config.epilogue_fusion_type =
        CutlassGemmConfig::EpilogueFusionType::FINALIZE;
}
```

## 总结

FC2 的 epilogue 可以完全融合 finalize 操作，包括：
1. **应用路由权重**：将专家输出乘以 softmax 权重
2. **Scatter 操作**：写回到原始 token 顺序
3. **多专家累加**：如果 K>1，累加多个专家的贡献
4. **偏置添加**：在 TP rank 0 添加偏置

这种融合显著提升了 MoE 的端到端性能，特别是在 Hopper (SM90+) 架构上。