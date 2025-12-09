# MoE Token 聚集（Token Gathering）详解

## 概述

Token 聚集是 MoE 中的关键操作，它将路由到同一专家的 token 收集到连续的内存位置，以便进行高效的批量 GEMM 计算。

## 1. 基本流程

### 原始状态
```
输入 Token 序列（按时间顺序）:
Token:   [T0, T1, T2, T3, T4, T5, T6, T7]
Expert:  [E1, E0, E2, E0, E1, E2, E0, E1]
         ↑路由器分配的专家ID
```

### Token 聚集后
```
重排序后的 Token（按专家分组）:
Expert 0: [T1, T3, T6]  ← 聚集到 Expert 0 的 token
Expert 1: [T0, T4, T7]  ← 聚集到 Expert 1 的 token
Expert 2: [T2, T5]      ← 聚集到 Expert 2 的 token
```

## 2. 具体实现步骤

### Step 1: 路由决策
```cpp
// 路由器输出每个 token 选择的专家
// router_logits: [num_tokens, num_experts]
// 经过 top-k softmax 后得到：
// - selected_experts: [num_tokens, k] - 每个 token 选择的 k 个专家
// - routing_weights: [num_tokens, k] - 对应的权重
```

### Step 2: 计算专家的 token 数量
```cpp
// 统计每个专家分配到的 token 数
int expert_token_counts[num_experts] = {0};

for (int token_idx = 0; token_idx < num_tokens; token_idx++) {
    for (int k = 0; k < experts_per_token; k++) {
        int expert_id = selected_experts[token_idx][k];
        expert_token_counts[expert_id]++;
    }
}

// 计算累积偏移（每个专家在重排序数组中的起始位置）
int expert_cumsum[num_experts + 1];
expert_cumsum[0] = 0;
for (int i = 0; i < num_experts; i++) {
    expert_cumsum[i + 1] = expert_cumsum[i] + expert_token_counts[i];
}
```

### Step 3: Token 重排序（Permutation）
```cpp
// 创建排序映射
int* permuted_row_to_unpermuted_row;  // 新位置 -> 原始位置
int* unpermuted_row_to_permuted_row;  // 原始位置 -> 新位置

// 临时计数器，用于跟踪每个专家当前的写入位置
int expert_write_idx[num_experts];
memcpy(expert_write_idx, expert_cumsum, sizeof(int) * num_experts);

// 执行重排序
for (int token_idx = 0; token_idx < num_tokens; token_idx++) {
    for (int k = 0; k < experts_per_token; k++) {
        int expert_id = selected_experts[token_idx][k];
        int new_position = expert_write_idx[expert_id]++;

        // 记录映射关系
        permuted_row_to_unpermuted_row[new_position] = token_idx;
        unpermuted_row_to_permuted_row[token_idx] = new_position;
    }
}
```

### Step 4: 数据聚集（Gather）
```cpp
// 实际的数据聚集操作
// 输入: hidden_states [num_tokens, hidden_size]
// 输出: permuted_data [expanded_num_tokens, hidden_size]

__global__ void gather_tokens_kernel(
    float* permuted_data,           // 输出：重排序后的数据
    const float* hidden_states,     // 输入：原始 token 数据
    const int* permuted_row_to_unpermuted_row,
    int num_tokens,
    int hidden_size)
{
    int permuted_idx = blockIdx.x;  // 新位置索引
    int feat_idx = threadIdx.x;     // 特征维度索引

    if (permuted_idx < num_tokens && feat_idx < hidden_size) {
        // 从原始位置读取数据
        int original_idx = permuted_row_to_unpermuted_row[permuted_idx];

        // 执行 gather 操作
        permuted_data[permuted_idx * hidden_size + feat_idx] =
            hidden_states[original_idx * hidden_size + feat_idx];
    }
}
```

## 3. 为什么需要 Token 聚集？

### 3.1 提高 GEMM 效率
```cpp
// 不聚集：需要为每个 token 单独调用 GEMM
for (int token_idx = 0; token_idx < num_tokens; token_idx++) {
    int expert_id = selected_experts[token_idx];
    // 单个 token 的 GEMM，效率低
    gemm_single_token(token_data[token_idx], expert_weights[expert_id], output[token_idx]);
}

// 聚集后：批量 GEMM，效率高
for (int expert_id = 0; expert_id < num_experts; expert_id++) {
    int start = expert_cumsum[expert_id];
    int end = expert_cumsum[expert_id + 1];
    int batch_size = end - start;

    if (batch_size > 0) {
        // 批量 GEMM：一次处理所有分配给该专家的 token
        batch_gemm(
            &permuted_data[start * hidden_size],  // 该专家的所有 token
            expert_weights[expert_id],             // 专家权重
            &expert_output[start * hidden_size],   // 输出
            batch_size,                           // M 维度
            hidden_size,                          // K 维度
            inter_size                            // N 维度
        );
    }
}
```

### 3.2 内存访问优化
- **连续内存访问**：同一专家的 token 在内存中连续存放
- **缓存友好**：减少缓存未命中
- **向量化加载**：可以使用向量加载指令

## 4. TensorRT-LLM 中的实现

### 相关数据结构
```cpp
// 在 moe_kernels.h 中
struct GroupedGemmInput {
    int64_t const* total_tokens_including_expert;  // 累积 token 数
    // total_tokens_including_expert[i] = 前 i 个专家的 token 总数
};

// 在执行时
int64_t* expert_first_token_offset;  // 每个专家的起始偏移
// expert_first_token_offset[i] = 专家 i 的第一个 token 在重排序数组中的位置
```

### Kernel 实现
```cpp
// 在 moe_util_kernels.cu 中
template <typename T>
__global__ void permute_tokens_kernel(
    T* permuted_output,
    const T* input,
    const int* permuted_indices,
    int num_tokens,
    int hidden_dim)
{
    // 高效的 coalesced 内存访问
    const int token_idx = blockIdx.x;
    const int hidden_idx = threadIdx.x;

    if (token_idx < num_tokens && hidden_idx < hidden_dim) {
        int src_token = permuted_indices[token_idx];
        permuted_output[token_idx * hidden_dim + hidden_idx] =
            input[src_token * hidden_dim + hidden_idx];
    }
}
```

## 5. Top-K 路由的额外复杂性

当 `experts_per_token > 1`（每个 token 选择多个专家）时：

```cpp
// 扩展的 token 数量
expanded_num_tokens = num_tokens * experts_per_token;

// 每个 token 会被复制 k 次，发送给 k 个不同的专家
for (int token_idx = 0; token_idx < num_tokens; token_idx++) {
    for (int k = 0; k < experts_per_token; k++) {
        int expert_id = selected_experts[token_idx][k];
        float weight = routing_weights[token_idx][k];

        // Token 被复制并加权
        weighted_token = input_token[token_idx] * weight;

        // 发送给对应专家
        send_to_expert(expert_id, weighted_token);
    }
}
```

## 6. 性能考虑

### 负载均衡
```cpp
// 理想情况：每个专家分配到相同数量的 token
ideal_tokens_per_expert = (num_tokens * experts_per_token) / num_experts;

// 实际情况：可能不均衡
// 需要负载均衡机制，如辅助损失（auxiliary loss）
```

### 内存开销
```cpp
// 需要额外的内存来存储：
// 1. 重排序后的数据
size_t permuted_data_size = expanded_num_tokens * hidden_size * sizeof(T);

// 2. 映射索引
size_t mapping_size = 2 * expanded_num_tokens * sizeof(int);

// 3. 专家偏移
size_t offset_size = (num_experts + 1) * sizeof(int64_t);
```

## 7. 融合优化（Fusion）

在 TensorRT-LLM 中，Token 聚集可以与其他操作融合：

```cpp
// 融合的 gather + 反量化
__global__ void fused_gather_dequantize_kernel(
    half* output,
    const int8_t* quantized_input,
    const float* scales,
    const int* gather_indices,
    int num_tokens,
    int hidden_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_tokens * hidden_dim) {
        int token_idx = idx / hidden_dim;
        int feat_idx = idx % hidden_dim;

        int src_token = gather_indices[token_idx];

        // 融合 gather 和反量化
        int8_t quant_val = quantized_input[src_token * hidden_dim + feat_idx];
        float scale = scales[src_token];
        output[idx] = __float2half(quant_val * scale);
    }
}
```

## 总结

Token 聚集是 MoE 实现中的关键操作，它：
1. **收集分配给同一专家的所有 token**
2. **重排序数据以实现连续内存访问**
3. **使能批量 GEMM 操作**
4. **可以与其他操作（如反量化）融合以提高效率**

这个操作是 MoE 能够高效执行的基础，将原本分散的 token 组织成适合 GPU 并行计算的形式。