# SID-lite / Free-MAD-lite smoke20 结果汇总

生成时间：2026-04-24  
Backbone：`dashscope/qwen-turbo-1101`  
数据集：`gsm8k`、`strategyqa`、`hotpotqa`，各 `smoke20_seed42`，共 60 题。

## 1. 运行产物

- SID-lite run：`local/runs/sid_lite/sid_lite_v1/smoke20/20260424T061208Z-sid_lite_v1-smoke20-dashscope-qwen-turbo-1101`
- SID-lite report：`local/reports/sid_lite/2026-04-24-sid_lite_v1-smoke20-dashscope-qwen-turbo-1101-report.md`
- Free-MAD-lite run：`local/runs/free_mad_lite/free_mad_lite_v1/smoke20/20260424T063104Z-free_mad_lite_v1-smoke20-dashscope-qwen-turbo-1101`
- Free-MAD-lite report：`local/reports/free_mad_lite/2026-04-24-free_mad_lite_v1-smoke20-dashscope-qwen-turbo-1101-report.md`

## 2. 验证状态

- SID-lite：`run_validation.json` 通过；60 个样本均包含 `mv_3`、`always_full`、`compression_only`、`sid_lite`；共享 Stage A hash、一致早退零通信、压缩包 token cap、非法 confidence fail-open 均通过。
- Free-MAD-lite：`run_validation.json` 通过；60 个样本均包含 `mv_3_initial`、`vanilla_mad_r1_final_vote`、`anti_conformity_final_vote`、`free_mad_lite_llm_trajectory`；单轮 debate、anti-conformity prompt hash、trajectory judge schema 均通过。
- API 调用：SID-lite 540/540 calls 完成，Free-MAD-lite 600/600 calls 完成；两者均无 request failure。

## 3. SID-lite 主结果

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens | Early Exit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6167 | 0.00 | 2271.30 | 3.00 | 0.271504 | 1.0000 |
| `always_full` | 0.6000 | 445.40 | 5219.10 | 6.00 | 0.114962 | 0.0000 |
| `compression_only` | 0.6500 | 220.03 | 4856.90 | 6.00 | 0.133830 | 0.0000 |
| `sid_lite` | 0.6500 | 76.77 | 3077.87 | 4.05 | 0.211185 | 0.6500 |

机制观察：

- `sid_lite` 在 smoke20 overall 上与 `compression_only` 准确率相同，通信 token 从 220.03 降到 76.77。
- `sid_lite` 相对 `always_full` 的 accuracy delta bootstrap CI 为 `[0.000000, 0.116667]`，仅作小样本方向性参考。
- 非法 confidence fail-open 发生 1 次，没有误早退。

## 4. Free-MAD-lite 主结果

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens | Judge Fallback |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mv_3_initial` | 0.6167 | 0.00 | 2034.03 | 3.00 | 0.303174 | 0.0000 |
| `vanilla_mad_r1_final_vote` | 0.7667 | 224.40 | 4537.10 | 6.00 | 0.168977 | 0.0000 |
| `anti_conformity_final_vote` | 0.6833 | 224.40 | 4627.93 | 6.00 | 0.147654 | 0.0000 |
| `free_mad_lite_llm_trajectory` | 0.7500 | 224.40 | 5625.85 | 7.00 | 0.133313 | 0.0000 |

机制观察：

- `free_mad_lite_llm_trajectory` 没有 judge fallback，trajectory judge schema 全部有效。
- 当前 smoke20 下，trajectory judge 明显优于 anti-conformity final vote，但略低于 vanilla final vote；相对 vanilla 的 accuracy delta CI 为 `[-0.083333, 0.033333]`。
- anti-conformity 在 GSM8K 上有收益，但在 StrategyQA 上出现负迁移，后续需要按数据集拆开分析。

## 5. 结论

SID-lite 的早退 + 压缩机制在本轮最值得继续放大：它保持或提升准确率，同时显著降低通信量和调用量。Free-MAD-lite 的轨迹裁决可用且稳定，但 anti-conformity prompt 本身不稳定，下一步更适合做 prompt ablation 或只把 trajectory judge 作为 vanilla debate 的聚合增强。

注意：两篇方法均按 lite 机制验证处理，不是 full reproduction；smoke20 结果仅用于工程联调和机制筛选。
