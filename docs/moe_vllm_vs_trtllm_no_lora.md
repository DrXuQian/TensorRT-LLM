# TensorRT‑LLM vs vLLM：MoE 路由/融合对比（不含 LoRA）

> 说明：本文只基于本仓库源码做“实现层面”的对比；所有关键结论用 `path:line` 标注来源，避免口径漂移。  
> 约束：**不含 LoRA**（TRTLLM MoE plugin 的 finalize fusion 也因此不会被 `hasLora()` 禁用：`cpp/tensorrt_llm/plugins/mixtureOfExperts/mixtureOfExpertsPlugin.cpp:339`）。

---

## 0. 术语对齐（两边名字对照）

- **路由输出**
  - TRTLLM：`token_selected_experts`（top‑k expert id）+ `token_final_scales`（top‑k 权重）。`tensorrt_llm/layers/moe.py:991`、`tensorrt_llm/layers/moe.py:1000`、`tensorrt_llm/layers/moe.py:1010`
  - vLLM：`topk_ids`（top‑k expert id）+ `topk_weights`（top‑k 权重）。`vllm/vllm/model_executor/layers/fused_moe/fused_moe.py:1146`
- **“展开后的 token 行”（expanded row index）**
  - vLLM `topkGatingSoftmax` 里会生成 `token_expert_indices`（源码里叫 `source_rows`），其定义是 `k_idx * num_rows + thread_row`：`vllm/csrc/moe/topk_softmax_kernels.cu:460`
  - vLLM 的 Triton `fused_moe_kernel` 使用 `offs_token // top_k` 回到原 token 行（从而在 GEMM 内“逻辑展开”输入）：`vllm/vllm/model_executor/layers/fused_moe/fused_moe.py:442`

---

## 1. TRTLLM（TRT engine）MoE：你列的 7 步在源码里的对应关系

### 1.1 路由（topkGatingSoftmax）

TRTLLM 路由发生在 TensorRT 网络图里：
- `default_routing`: `softmax(cast(logits,float32)) -> topk`：`tensorrt_llm/layers/moe.py:893`
- `renormalize`: `topk(cast(logits,float32)) -> softmax(topk_scores)`：`tensorrt_llm/layers/moe.py:900`

### 1.2 MoE plugin 内部 pipeline（permute/sort → expandInput → GEMM1 → activation → GEMM2 → finalize）

MoE plugin 的 runner 主入口：`CutlassMoeFCRunner::runMoe(...)`：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:3724`

按“概念步骤”对齐如下：
1) **permute/sort（prologue，构建/排序映射表）**
   - 优先尝试 fused：`fusedBuildExpertMapsSortFirstToken(...)`：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:3788`
   - 不满足条件则回退：`threeStepBuildExpertMapsSortFirstToken(...)`：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:3796`
2) **expandInput**
   - `expandInputRowsKernelLauncher(...)`：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:3824`
3) **FC1 GEMM（GEMM1）**
   - `Self::gemm1(...)` 调用点：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:3869`
4) **SwiGLU 等 activation**
   - 若走 TMA WS（SM90+）路径：GEMM1 之后显式 `doActivation(...)`：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:3144`
   - 若非 TMA WS 且 gated activation 允许 fuse：`use_ampere_activation_fusion = isFusedGatedActivation(...)`：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:3087`，判定条件见 `supportsFusedGatedActivation(...)`：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_template_dispatch.h:652`
5) **FC2 GEMM（GEMM2）**
   - `Self::gemm2(...)` 调用点：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:3895`
6) **finalize routing（unpermute + top‑k reduce/scale + bias…）**
   - 默认：`finalizeMoeRoutingKernelLauncher(...)`：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:3361`
   - 若 GEMM2 选择了 `EpilogueFusionType::FINALIZE` 且 runner 允许：设置 `gemm2_tma_ws_input.fusion = FINALIZE` 并 `setFinalizeFusionParams(...)`：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:4058`

> TRTLLM MoE plugin 的“融合矩阵 + 触发条件（按架构/量化/形状）”建议直接看：`docs/moe_fusion_trt_engine_no_lora.md:1`。

---

## 2. vLLM（torch fused_moe）：你列的 7 步在源码里的对应关系

### 2.1 共同路由：topkGatingSoftmax（topk + softmax + 可选 renormalize）

调用链：
- `fused_topk(...)` 分配 `topk_weights/topk_ids/token_expert_indices` 并调用 `topk_func(...)`：`vllm/vllm/model_executor/layers/fused_moe/fused_moe.py:1146`
- `vllm_topk_softmax(...)` → `ops.topk_softmax(...)`：`vllm/vllm/model_executor/layers/fused_moe/fused_moe.py:1120`
- `ops.topk_softmax` → `torch.ops._moe_C.topk_softmax(...)`：`vllm/vllm/_custom_ops.py:2109`
- CUDA 实现入口 `topk_softmax(...)`：`vllm/csrc/moe/topk_softmax_kernels.cu:675`

关键细节（对后续“permute/unpermute”很重要）：
- `token_expert_indices`（源码变量名 `source_rows`）写入规则：`source_rows[idx] = k_idx * num_rows + thread_row`：`vllm/csrc/moe/topk_softmax_kernels.cu:460`
- `renormalize=True` 时会把选中 top‑k 权重除以 `selected_sum`：`vllm/csrc/moe/topk_softmax_kernels.cu:482`

### 2.2 vLLM 的 3 条“专家侧”实现路径（你看到的 step list 属于其中之一）

#### A) TritonExperts（Triton fused MoE GEMM kernel）

这一条不走 `_count_expert_num_tokens` / `ep_scatter_*`，而是走 `moe_align_block_size`：
- 排序/对齐：`sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(...)`：`vllm/vllm/model_executor/layers/fused_moe/fused_moe.py:2144`
- `moe_align_block_size` 本身是 C++/CUDA op：`vllm/vllm/model_executor/layers/fused_moe/moe_align_block_size.py:11`，其核心语义写在 docstring（含 EP 的 expert_map 处理）：`vllm/vllm/model_executor/layers/fused_moe/moe_align_block_size.py:23`
- Triton GEMM 内部“逻辑展开输入”的关键：从 `offs_token` 回到原 token 行用 `offs_token // top_k`：`vllm/vllm/model_executor/layers/fused_moe/fused_moe.py:442`
- GEMM2 可在 kernel 内 fuse `mul_routed_weight`：`vllm/vllm/model_executor/layers/fused_moe/fused_moe.py:523`
- 激活（SwiGLU/GELU‑GLU 等）走 `torch.ops._C.silu_and_mul(...)` 等：`vllm/vllm/model_executor/layers/fused_moe/modular_kernel.py:574`，对应 CUDA kernel：`vllm/csrc/activation_kernels.cu:22`
- 最终 reduce：`ops.moe_sum(...)`：`vllm/vllm/model_executor/layers/fused_moe/fused_moe.py:2213`（C++ 实现里对 topk=2/3/4 有专门 kernel：`vllm/csrc/moe/moe_align_sum_kernels.cu:610`）

#### B) DeepGemmExperts（EP scatter/gather + DeepGemm grouped GEMM）

这条最接近你列的 vLLM 步骤（2‑4 是 triton scatter/gather）：
- DeepGemm 是否启用：`_valid_deep_gemm(...)`（包含对齐/shape、dtype、contiguous、以及 `N > 512` 等限制）：`vllm/vllm/model_executor/layers/fused_moe/deep_gemm_moe.py:44`
- 选择 deepgemm 还是 triton：`TritonOrDeepGemmExperts.apply`：`vllm/vllm/model_executor/layers/fused_moe/triton_deep_gemm_moe.py:140`
- 计数（对应你 list 的 `_count_expert_num_tokens`）：`count_expert_num_tokens(...)`：`vllm/vllm/model_executor/layers/fused_moe/utils.py:63`
- scatter 第 1 段（对齐每个 expert 的 token 数到 128）：`tokens_per_expert = round_up_128(tokens_per_expert)`：`vllm/vllm/model_executor/layers/fused_moe/deep_gemm_utils.py:77`
- scatter 第 2 段（按 expert 做 `atomic_add` 分配目标行，并写 `output_index` / 拷贝输入）：`dest_token_index = tl.atomic_add(...)`：`vllm/vllm/model_executor/layers/fused_moe/deep_gemm_utils.py:146`
- GEMM1 / GEMM2：`m_grouped_fp8_gemm_nt_contiguous(...)`：`vllm/vllm/model_executor/layers/fused_moe/deep_gemm_moe.py:253`
- activation：
  - 部分 scale 格式下会用 Triton 的“SiLU+mul+quant”融合 kernel：`silu_mul_per_token_group_quant_fp8_colmajor(...)`：`vllm/vllm/model_executor/layers/fused_moe/deep_gemm_moe.py:183`
  - 否则可能先走 `torch.ops._C.silu_and_mul` 再做量化：`vllm/vllm/model_executor/layers/fused_moe/deep_gemm_moe.py:172`
- finalize routing（unpermute + topk weighted reduce）：`ep_gather(...)`：`vllm/vllm/model_executor/layers/fused_moe/deep_gemm_utils.py:291`

> 你提到的 `_fwd_kernel_ep_scatter_2_bf16` 在本仓库 vLLM 目录里没有同名符号（`rg` 无匹配）；这里对应的是通用的 `_fwd_kernel_ep_scatter_2`，数据类型由传入 tensor 决定：`vllm/vllm/model_executor/layers/fused_moe/deep_gemm_utils.py:94`。

#### C) CutlassExperts（moe_permute/unpermute + CUTLASS grouped GEMM）

这条更接近 TRTLLM “permute/sort → expandInput → gemm → finalize”的形态：
- `moe_permute`/`moe_unpermute` 是 `_moe_C` 自定义 op：`vllm/csrc/moe/torch_bindings.cpp:90`
- 其 CUDA 实现要求 CUDA >= 12：`vllm/csrc/moe/moe_permute_unpermute_op.cu:8`
- `moe_permute` 内部做了：
  - EP 预处理 topk id：`preprocessTopkIdLauncher(...)`：`vllm/csrc/moe/moe_permute_unpermute_op.cu:65`
  - sort+scan 得到 `expert_first_token_offset`：`sortAndScanExpert(...)`：`vllm/csrc/moe/moe_permute_unpermute_op.cu:69`
  - expand input：`expandInputRowsKernelLauncher(...)`：`vllm/csrc/moe/moe_permute_unpermute_op.cu:77`
- `moe_unpermute` 内部 finalize：`finalizeMoeRoutingKernelLauncher(...)`：`vllm/csrc/moe/moe_permute_unpermute_op.cu:122`
- cutlass FP8 MoE 的 Python glue（含 `ops.cutlass_moe_mm` / activation / quant / `moe_unpermute`）：`vllm/vllm/model_executor/layers/fused_moe/cutlass_moe.py:176`

---

## 3. 对比矩阵（不含 LoRA）：步骤 ↔ 实现 / 融合点

> 表里“融合”只指“在同一个 kernel / op 内完成”，不讨论框架/编译器可能做的额外图优化。

| 概念步骤 | TRTLLM（TRT engine, MoE plugin） | vLLM TritonExperts | vLLM DeepGemmExperts | vLLM CutlassExperts |
|---|---|---|---|---|
| 1. topk+softmax | TRT 图算子：`default_routing` / `renormalize`：`tensorrt_llm/layers/moe.py:893` | `fused_topk` → `_moe_C.topk_softmax`：`vllm/vllm/model_executor/layers/fused_moe/fused_moe.py:1146` | 同左（路由通用） | 同左（路由通用） |
| 2. permute/sort | prologue：`fusedBuild...` 或 `threeStepBuild...`：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:3788` | `moe_align_block_size`（C++/CUDA）：`vllm/vllm/model_executor/layers/fused_moe/moe_align_block_size.py:11` | `count_expert_num_tokens` + `ep_scatter_1/2`：`vllm/vllm/model_executor/layers/fused_moe/deep_gemm_utils.py:60` | `moe_permute` 内部 `preprocess+sort+scan`：`vllm/csrc/moe/moe_permute_unpermute_op.cu:69` |
| 3. expandInput | `expandInputRowsKernelLauncher`：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:3824` | **不单独 materialize**：GEMM 内 `offs_token//top_k` 逻辑展开：`vllm/vllm/model_executor/layers/fused_moe/fused_moe.py:442` | `ep_scatter_2` 直接写出 per‑expert contiguous buffer：`vllm/vllm/model_executor/layers/fused_moe/deep_gemm_utils.py:151` | `moe_permute` 内部 `expandInputRowsKernelLauncher`：`vllm/csrc/moe/moe_permute_unpermute_op.cu:77` |
| 4. GEMM1 | plugin 内 `Self::gemm1`：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:3869` | `invoke_fused_moe_kernel(..., top_k=topk)`：`vllm/vllm/model_executor/layers/fused_moe/fused_moe.py:544` | `m_grouped_fp8_gemm_nt_contiguous`：`vllm/vllm/model_executor/layers/fused_moe/deep_gemm_moe.py:253` | `ops.cutlass_moe_mm`：`vllm/vllm/model_executor/layers/fused_moe/cutlass_moe.py:197` |
| 5. SwiGLU activation | TMA WS 路径通常单独 `doActivation`：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:3144` | `torch.ops._C.silu_and_mul` → `act_and_mul_kernel`：`vllm/vllm/model_executor/layers/fused_moe/modular_kernel.py:579` | 可能 fused（SiLU+mul+quant）：`vllm/vllm/model_executor/layers/fused_moe/deep_gemm_moe.py:183` | 同 TritonExperts：`vllm/vllm/model_executor/layers/fused_moe/modular_kernel.py:579` |
| 6. GEMM2 | plugin 内 `Self::gemm2`：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:3895` | `invoke_fused_moe_kernel(..., MUL_ROUTED_WEIGHT=...)`：`vllm/vllm/model_executor/layers/fused_moe/fused_moe.py:523` | `m_grouped_fp8_gemm_nt_contiguous`：`vllm/vllm/model_executor/layers/fused_moe/deep_gemm_moe.py:265` | `ops.cutlass_moe_mm`：`vllm/vllm/model_executor/layers/fused_moe/cutlass_moe.py:221` |
| 7. finalize routing（unpermute+reduce） | 默认 `finalizeMoeRoutingKernelLauncher`；或 GEMM2 FINALIZE fusion：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:3361`、`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:4058` | `ops.moe_sum`（topk 维求和，权重可在 GEMM2 内已乘）：`vllm/vllm/model_executor/layers/fused_moe/fused_moe.py:2213` | `ep_gather`（带 topk_weight accumulate）：`vllm/vllm/model_executor/layers/fused_moe/deep_gemm_utils.py:280` | `moe_unpermute` → `finalizeMoeRoutingKernelLauncher`：`vllm/csrc/moe/moe_permute_unpermute_op.cu:122` |

---

## 4. 额外说明：为什么“TRT engine 可能更快”（但要看条件）

只从实现角度（不做跑分承诺）：
- TRTLLM 把 **prologue / expand / GEMM / finalize** 放在一个 plugin 调度域内（同一个 `runMoe`），并且在 SM90+ 上存在把 finalize 合入 GEMM2 epilogue 的路径：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:4058`。
- vLLM 的 torch 路径通常由多个 custom op / triton kernel 组成（例如 `moe_align_block_size`、两次 GEMM kernel、activation kernel、`moe_sum`），调度上更“分散”：`vllm/vllm/model_executor/layers/fused_moe/fused_moe.py:2144`、`vllm/vllm/model_executor/layers/fused_moe/fused_moe.py:2213`。

是否更快取决于：硬件（是否 SM90+ 等）、量化形态、batch/seq、top‑k、EP/TP、以及是否满足各自“融合”分支的前置条件（详见 `docs/moe_fusion_trt_engine_no_lora.md:1`）。

