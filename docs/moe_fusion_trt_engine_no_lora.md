# TensorRT‑LLM：TRT Engine（MoE plugin / OOTB）融合矩阵（不含 LoRA）

本文档只覆盖 **TRT engine** 路径（即构建 TensorRT engine 时 MoE 层如何实现/融合），并假设 **未启用 LoRA**。

> 说明：所有结论都以仓库源码为准，文中用 `path:line` 标注直接来源位置。

---

## 1. 走哪条实现路径：MoE plugin vs MoeOOTB

### 1.1 `--moe_plugin` 如何影响实现

- `PluginConfig.moe_plugin` 默认值为 `"auto"`：`tensorrt_llm/plugin/plugin.py:184`。
- CLI 传入的 `"disable"` 会被映射为 `None`（对 Optional dtype plugin）：`tensorrt_llm/plugin/plugin.py:323`。
- 构建时用 `use_ootb_moe = (build_config.plugin_config.moe_plugin is None)` 判定是否改用 OOTB：`tensorrt_llm/builder.py:788`。
- `use_ootb_moe=True` 时会把 `MOE` 层替换成 `MoeOOTB`：`tensorrt_llm/models/modeling_utils.py:1635`、`tensorrt_llm/models/modeling_utils.py:1510`。

### 1.2 MoeOOTB 的关键限制（影响可用融合/量化形态）

- MoeOOTB 不支持 weight‑only quant：`tensorrt_llm/layers/moe.py:1229`。
- MoeOOTB 不支持 `SM >= 100`：`tensorrt_llm/layers/moe.py:1234`。

---

## 2. TRT `MixtureOfExperts` plugin：dtype / 架构硬约束（不含 LoRA）

### 2.1 输出 dtype 与输入 dtype

- 只有当输入 dtype 为 FP8/FP4 时，才允许 `output_type != input_type`；否则必须相同：`cpp/tensorrt_llm/plugins/mixtureOfExperts/mixtureOfExpertsPlugin.cpp:258`。

### 2.2 架构下限

- FP8：要求 `SM >= 89`：`cpp/tensorrt_llm/plugins/mixtureOfExperts/mixtureOfExpertsPlugin.cpp:260`。
- FP4：要求 `SM >= 100`：`cpp/tensorrt_llm/plugins/mixtureOfExperts/mixtureOfExpertsPlugin.cpp:262`。

### 2.3 直接输出 FP8/FP4（当前不支持）

- `switch_output_type` 中对 `output_type == kFP8/kFP4` 直接抛错（注释也说明缺少 FP8 reduction 支持）：`cpp/tensorrt_llm/plugins/mixtureOfExperts/mixtureOfExpertsPlugin.cpp:227`、`cpp/tensorrt_llm/plugins/mixtureOfExperts/mixtureOfExpertsPlugin.cpp:230`。

---

## 3. TRT engine（MoE plugin 路径）融合矩阵（不含 LoRA）

下面的融合都发生在 `CutlassMoeFCRunner::runMoe(...)`（MoE plugin 内部 runner 调用的实现）：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:3782`。

### 3.1 路由 Prologue 融合（构建/排序 expert maps）

| 子步骤 | 融合实现 | 触发条件（全部满足） | 回退实现 |
|---|---|---|---|
| Prologue：`token_selected_experts` → `permuted_row_to_unpermuted_row` / `unpermuted_row_to_permuted_row` / `expert_first_token_offset` | `fusedBuildExpertMapsSortFirstToken(...)` | runner 只在 `!use_wfp4a16` 时尝试 fused prologue：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:3785`；且该函数必须返回 `true`（否则回退）：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:3793`。返回 `false` 的硬条件包括：`num_tokens <= 256`（否则不支持）：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:477`；`experts_per_token ∈ {1,2,4,6,8}`（否则不支持）：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:510`、`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:539`；`expert_log = floor(log2(num_experts_per_node+1))+1 <= 9`（否则不支持）：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:555`、`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:568`；single‑block 要求：`blocks == 1`：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:432`；动态 smem 必须小于设备上限（否则返回 false）：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:453`、`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:454`。 | `threeStepBuildExpertMapsSortFirstToken(...)`：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:3796` |

### 3.2 ExpandInputRows（expandInput）

ExpandInputRows 是 MoE plugin 内部的一个独立 kernel，用于把输入按路由结果“复制 + 重排”为按 expert 分块的连续布局，并同步重排路由权重（scales）等辅助张量。

- 调用点：prologue 之后、GEMM1 之前固定调用 `expandInputRowsKernelLauncher(...)`：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:3824`。
- 内核语义：注释明确这是“Duplicated and permutes rows for MoE”，并解释了 `permuted_row_to_unpermuted_row` 在 expanded index 空间上的含义：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:1368`、`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:1374`。
- 该 kernel 同时处理 `unpermuted_scales -> permuted_scales`（路由权重重排）：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:1384`。
- **AWQ prequant scale 融合到 expandInput**：kernel 模板参数 `PRE_QUANT_AWQ` 与 block scaling 互斥（静态断言）：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:1391`。运行时 `runMoe` 只在 `(use_w4afp8 && !use_fp8_input)` 时把 `quant_params.groupwise.fc1.act_scales` 作为 `prequant_scales` 传入（否则传 `nullptr`）：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:3828`。
- `use_per_expert_act_scale` 目前仅在 NVFP4xNVFP4 场景启用，并作为参数传入 expandInput：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:3820`、`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:3826`。

### 3.3 FC1（GEMM1）与激活/门控融合

| 子步骤 | 融合实现 | 触发条件（全部满足） | 不满足时 |
|---|---|---|---|
| FC1 非 gated 激活（relu/gelu/silu/identity 等） | `moeGemmBiasAct`（GEMM + bias + activation epilogue） | 发生在非 gated 分支：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:3186`，最终调用 `moeGemmBiasAct`：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:3208`。激活类型 → epilogue 映射见 `moeGemmBiasAct` 的 switch：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_template_dispatch.h:965`。 | 不适用（该行描述的就是“能 fuse 的”情况） |
| FC1 gated 激活（Swiglu/Geglu）在 **SM80 fallback** 上的融合 | “Ampere fused gated activation” 内核（`use_fused_moe=true` 触发 `sm80_generic_fused_moe_gemm_kernelLauncher`） | 判定入口：`use_ampere_activation_fusion = gemm_runner.isFusedGatedActivation(...)`：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:3087`。精确条件：`activation ∈ {Swiglu,Geglu}`、`T==WeightType`、`T!=float`、`!use_fp8`、`SM>=80`、`gemm_k%64==0 && gemm_n%64==0`：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_template_dispatch.h:656`；并要求 `!gemm_config.is_tma_warp_specialized`：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_template_dispatch.h:665`。底层 fused kernel 额外限制：A/B 都是 2‑byte 且 epilogue tag 为 Silu/FtGelu：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_template_dispatch.h:202`。 | 先用 `ActivationType::Identity` 跑 GEMM1，再单独 `doGatedActivation(...)`：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:3224`、`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:3249` |
| FC1 在 **TMA warp specialized（SM90+/Blackwell）** 路径的融合能力 | 仅支持 `fusion = NONE/FINALIZE`（不支持 activation/gated activation fusion） | TMA WS dispatch 里对 `fusion == ACTIVATION/GATED_ACTIVATION` 直接 `TLLM_THROW("Unimplemented fusion")`：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_template_dispatch.h:774`。因此 TMA WS 路径（SM90+）里，FC1 激活通常通过独立 `doActivation(...)` 完成（例如 `using_tma_ws_gemm1` 分支：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:3101`、随后 `doActivation`：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:3152`）。 | 不适用 |

### 3.4 AWQ：将 FC2 prequant scale 融合进 FC1 激活（仅 gated）

| 子步骤 | 融合实现 | 触发条件（全部满足） | 不满足时 |
|---|---|---|---|
| FC2 prequant scale（用于 GEMM2 前的 act scale） | 通过把 `fc2_prequant_scale_ptr` 传给 `gemm1(...)`，在 FC1 激活阶段应用 | `fuse_fc2_prequant_scale = use_awq && is_gated_activation`：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:3865`；其中 `use_awq` 精确定义为：`quant_params.groupwise.fc1.act_scales && quant_params.groupwise.fc2.act_scales && !use_wfp4a16`：`cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h:810`。启用时 `fc2_prequant_scale_ptr` 传入 `gemm1(...)`：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:3874`；并由 gated activation kernel 的 `prequant_scale` 参数实际应用：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:1969`、`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:2028`。 | `applyPrequantScale(... fc2 ...)` 在 GEMM2 前单独执行：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:3889` |

### 3.5 FC2（GEMM2）+ Finalize 融合（Finalize fusion）

Finalize fusion 指把 “finalize（unpermute + top‑k reduction/scale + bias 等）” 合入 GEMM2 的 TMA WS epilogue，从而不再调用 `finalizeMoeRoutingKernelLauncher(...)`。

| 子步骤 | 融合实现 | 触发条件（全部满足） | 不满足时 |
|---|---|---|---|
| FC2（GEMM2）+ finalize | `gemm2_tma_ws_input.fusion = FINALIZE` 并设置 `setFinalizeFusionParams(...)` | 1) 必须具备 TMA WS 支持：`supportsTmaWarpSpecialized(sm)` 仅对 `sm==90`、`100<=sm<120`、`sm==120||121` 返回真（且要求 isValid*Specialisation）：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_template_dispatch.h:635`。2) tactic 生成侧：`getTactics(GEMM_2)` 只有在 `mayHaveFinalizeFused()` 为真时才会把 `supports_finalize_fusion=true` 传给 `getConfigs`：`cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h:614`；`mayHaveFinalizeFused()` 的条件是 `supportsTmaWarpSpecialized() && SM>=90 && use_fused_finalize_ && !use_wfp4a16`：`cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h:827`。3) plugin 侧 `use_fused_finalize_` 的赋值：`(experts_per_token < 3 || !mUseDeterministicKernels) && !TRTLLM_MOE_DISABLE_FINALIZE_FUSION && !hasLora()`：`cpp/tensorrt_llm/plugins/mixtureOfExperts/mixtureOfExpertsPlugin.cpp:338`（环境变量 `TRTLLM_MOE_DISABLE_FINALIZE_FUSION`：`cpp/tensorrt_llm/common/envUtils.cpp:422`）。4) 运行时必须实际选中 `gemm2_config_->epilogue_fusion_type == FINALIZE`：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:4058`；并通过一致性检查：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:4061`；最终设置 `fusion=FINALIZE`：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:4067`。 | 走完 GEMM2 后调用 `finalizeMoeRoutingKernelLauncher(...)`：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:3361` |

---

## 4. “不含 LoRA”下的 finalize fusion 实用化简（从源码直接推导）

在 TRT MoE plugin 场景中，`use_fused_finalize_` 的赋值包含 `!hasLora()`：`cpp/tensorrt_llm/plugins/mixtureOfExperts/mixtureOfExpertsPlugin.cpp:338`。因此本文件假设 **无 LoRA** 时，这一项恒为真；最终是否启用 finalize fusion 主要受以下三类条件控制：

1) **top_k / determinism**：`experts_per_token < 3 || !mUseDeterministicKernels`：`cpp/tensorrt_llm/plugins/mixtureOfExperts/mixtureOfExpertsPlugin.cpp:338`。  
2) **环境变量硬禁用**：`TRTLLM_MOE_DISABLE_FINALIZE_FUSION`：`cpp/tensorrt_llm/common/envUtils.cpp:422`。  
3) **架构与类型组合**：必须走 TMA WS（SM90+/Blackwell）并且 `!use_wfp4a16`：`cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h:827`、`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_template_dispatch.h:635`。
