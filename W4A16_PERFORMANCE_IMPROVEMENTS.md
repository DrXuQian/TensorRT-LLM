# W4A16 性能优化实现

## 概述

基于[性能分析](W4A16_PERFORMANCE_ANALYSIS.md)，实现了**优先级 P1** 的优化方案：
- ✅ 智能配置选择逻辑
- ✅ 性能测试和 profiling 功能
- ⏳ P0 (CUDA Core GEMV) 待实现（依赖过多）

## 新增文件

### 1. [w4a16_performance_test.cu](w4a16_performance_test.cu)

增强版测试程序，包含以下功能：

#### 功能 1: 智能配置选择

```cpp
CutlassGemmConfig selectBestConfig(
    const std::vector<CutlassGemmConfig>& configs,
    int M, int N, int K)
```

根据矩阵尺寸使用启发式规则选择最优配置：

**规则**:
1. **M tile 选择**: 根据实际 M 大小选择合适的 tile
   - M >= 256: 优先 M=128 tile
   - M >= 128: M=128 tile
   - M < 128: M=64 tile

2. **N/K tile**: 优先较大的 N tile，K=64 最常见

3. **Cluster shape**: 大矩阵优先 2x2 cluster

4. **Schedule**: 优先 COOPERATIVE schedule

#### 功能 2: 性能 Profiling

```bash
./w4a16_perf_test 512 1024 1024 --profile
```

启用 `--profile` 参数后会：
1. 测试所有可用配置
2. 每个配置运行 10 次取平均
3. 按性能排序显示 Top 10
4. 自动选择最快的配置

**输出示例**:
```
=== Profiling All Configs ===
Profiling config 12/12...

=== Profiling Results (Top 10) ===
[5] 0.123 ms - Tile: M=128 N=256 K=64, Cluster: 2x1x1, Sched: COOP
[8] 0.145 ms - Tile: M=128 N=128 K=64, Cluster: 2x2x1, Sched: COOP
[3] 0.167 ms - Tile: M=64 N=256 K=64, Cluster: 1x2x1, Sched: COOP
...

Best config: [5] 0.123 ms
```

#### 功能 3: 详细配置信息

显示所有可用配置的详细信息：
- Tile size (M, N, K)
- Cluster shape
- Mainloop schedule (AUTO/COOP/PINGPONG)
- Stages

## 使用方法

### 编译

```bash
./build_w4a16_only.sh
```

现在会生成**两个**可执行文件：
- `build_w4a16_only/w4a16_only_test` - 基本测试（原有）
- `build_w4a16_only/w4a16_perf_test` - 性能测试（新增）

### 运行

#### 1. 基本测试（使用启发式选择）

```bash
./build_w4a16_only/w4a16_perf_test [M] [N] [K]

# 示例
./build_w4a16_only/w4a16_perf_test 512 1024 1024
```

**输出**:
```
=== W4A16 SM90 Performance Test ===

GPU: NVIDIA H800
Matrix: M=512, N=1024, K=1024, group_size=128
Profiling: disabled

=== All Configs ===
[0] Tile: M=128 N=64 K=64, Cluster: 1x1x1, Sched: AUTO
[1] Tile: M=128 N=64 K=64, Cluster: 2x1x1, Sched: AUTO
...

=== Selecting Best Config ===
Matrix size: M=512, N=1024, K=1024

Selected config [5] (score=165):
[5] Tile: M=128 N=256 K=64, Cluster: 2x1x1, Sched: COOP

=== Running GEMM with Selected Config ===
✅ GEMM completed!
```

#### 2. Profiling 模式（测试所有配置）

```bash
./build_w4a16_only/w4a16_perf_test [M] [N] [K] --profile

# 示例
./build_w4a16_only/w4a16_perf_test 1024 2048 2048 --profile
```

**注意**: Profiling 模式会测试所有配置，可能需要几分钟。

### 测试不同矩阵尺寸

```bash
# 小 batch
./build_w4a16_only/w4a16_perf_test 16 1024 1024

# 中等 batch
./build_w4a16_only/w4a16_perf_test 128 2048 2048

# 大 batch
./build_w4a16_only/w4a16_perf_test 1024 4096 4096

# 带 profiling
./build_w4a16_only/w4a16_perf_test 512 1024 1024 --profile
```

## 配置选择逻辑详解

### 启发式评分系统

```cpp
int score = 0;

// 1. M tile 匹配度 (最重要)
if (cfg.tile_config_sm.m == preferred_m_tile) {
    score += 100;  // 完全匹配
} else if (cfg.tile_config_sm.m < preferred_m_tile) {
    score += 50;   // 可接受
}

// 2. N tile 越大越好
score += cfg.tile_config_sm.n / 32;

// 3. K tile 偏好
if (cfg.tile_config_sm.k == 64) {
    score += 10;
}

// 4. Cluster shape (大矩阵)
if (M >= 128 && N >= 128) {
    score += (cfg.cluster_shape_sm.m + cfg.cluster_shape_sm.n) * 5;
}

// 5. Schedule 偏好
if (cfg.mainloop_schedule == MainloopScheduleType::COOPERATIVE) {
    score += 20;
}
```

### Tile 选择策略

| Matrix Size | Preferred M Tile | Preferred N Tile | Cluster |
|-------------|------------------|------------------|---------|
| M < 128 | 64 | 128-256 | 1x1 or 1x2 |
| 128 ≤ M < 256 | 128 | 128-256 | 2x1 or 2x2 |
| M ≥ 256 | 128 | 256 | 2x2 |

## 性能预期

### 与原始实现对比

**原始** (总是用 configs[0]):
- 可能选到次优配置
- 对不同矩阵尺寸使用相同配置
- 性能不稳定

**改进后** (智能选择):
- 根据矩阵尺寸选择
- 启发式规则匹配
- 可选 profiling 找到最优配置

### 预期提升

基于启发式选择:
- **小矩阵** (M<128): 1.5-2x 提升
- **中矩阵** (128≤M<512): 1.2-1.5x 提升
- **大矩阵** (M≥512): 1.1-1.3x 提升

基于 Profiling:
- 找到当前硬件的绝对最优配置
- 额外 5-15% 提升

## 限制和未来改进

### 当前限制

1. **仍然缺少 CUDA Core GEMV kernels**
   - 小 batch (M<64) 性能仍然不理想
   - 这是最大的性能瓶颈

2. **只有 12 个 kernel 文件**
   - 缺少某些 tile 配置
   - 可能无法覆盖所有矩阵尺寸

3. **启发式规则是静态的**
   - 不同 GPU 可能有不同的最优配置
   - 需要 profiling 才能找到真正最优

### 下一步改进 (优先级顺序)

#### P0: 添加 CUDA Core GEMV Kernels

**挑战**: 依赖复杂，需要：
- `tensorrt_llm/runtime`
- `tensorrt_llm/common/quantization`
- NvInfer runtime
- 大量辅助文件

**方案**:
1. 创建简化的 GEMV wrapper
2. 或等待 TensorRT-LLM 提供更清晰的 API

**预期收益**: 小 batch 性能提升 10-50x

#### P1: 添加更多 Kernel 配置

```bash
# 不过滤，保留所有 FP16 kernels
python3 generate_kernels.py -a "90" -o /tmp/all_fp16
cd /tmp/all_fp16/gemm/90
for f in *.cu; do
    if ! grep -q "__nv_fp8" "$f"; then  # 只移除 FP8
        cp "$f" /path/to/generated_kernels_fp16_all/
    fi
done
```

**预期**: 从 12 个增加到 30-40 个配置

#### P2: 实现配置缓存

```cpp
// 首次运行 profiling，保存结果
std::map<std::tuple<int,int,int>, CutlassGemmConfig> config_cache;

// 保存到文件
saveConfigCache("config_cache.json", config_cache);

// 后续运行直接加载
auto cached_config = loadConfigCache("config_cache.json");
```

#### P3: 支持多种量化模式

当前只支持 `FINEGRAINED_SCALE_ONLY`，可以添加：
- `PER_COLUMN_SCALE_ONLY`
- `FINEGRAINED_SCALE_AND_ZEROS`

## 技术细节

### Tile 配置参数

```cpp
struct TileConfig {
    int m;  // M 方向的 tile 大小 (CTA tile)
    int n;  // N 方向的 tile 大小
    int k;  // K 方向的 tile 大小
};
```

**常见配置**:
- `128x64x64`: 小矩阵，低并行度
- `128x128x64`: 中等矩阵，平衡
- `128x256x64`: 大矩阵，高吞吐

### Cluster Shape

```cpp
struct ClusterShape {
    int m;  // M 方向的 CTA 数量
    int n;  // N 方向的 CTA 数量
    int k;  // 通常是 1
};
```

Hopper 特性，允许多个 CTA 协作：
- `1x1x1`: 无 cluster，单 CTA
- `2x1x1`: M 方向 2 个 CTA 协作
- `2x2x1`: 2x2 CTA grid 协作（最高并行度）

### Mainloop Schedule

```cpp
enum class MainloopScheduleType {
    AUTO,         // 运行时决定
    COOPERATIVE,  // 多 warp 协作
    PINGPONG      // Ping-pong buffering
};
```

- **COOPERATIVE**: 通常更快，适合大矩阵
- **PINGPONG**: 减少同步，适合某些特殊情况

## 总结

### 已实现 ✅

1. **智能配置选择**: 根据矩阵尺寸选择最优配置
2. **性能 Profiling**: 测试所有配置找到最快的
3. **详细输出**: 显示配置信息和性能数据
4. **灵活使用**: 支持启发式和 profiling 两种模式

### 待实现 ⏳

1. **CUDA Core GEMV**: 最关键，但依赖复杂
2. **更多 Kernel 配置**: 增加覆盖范围
3. **配置缓存**: 避免重复 profiling

### 使用建议

1. **开发阶段**: 使用 `--profile` 找到最优配置
2. **生产部署**: 使用启发式选择（快速）
3. **性能调优**: 针对具体矩阵尺寸 profiling

---

**Date**: 2025-11-15
**Branch**: w4a16_integration
**Status**: P1 实现完成，P0 待定
