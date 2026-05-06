# HotpotQA 通信必要性 pilot100 报告

## 1. 实验概览

- 实验名：`hotpotqa_split500_main`
- Phase：`pilot100`
- Backbone：`xiaomimimo/mimo-v2.5`
- 运行目录：`runs/comm_necessary/hotpotqa_split500_main/pilot100/20260506T140709Z-hotpotqa_split500_main-pilot100-xiaomimimo-mimo-v2.5`
- 任务：HotpotQA split-context evidence exchange；当前报告用于 `pilot100` 阶段的工程验证与结果汇总。
- 方法：`full_context_single`、`split_no_comm_mv3`、`answer_only_exchange`、`evidence_exchange`、`full_packet_exchange`。

## 2. 主结果表

| Method | Ans EM | Ans F1 | Sup F1 | Joint F1 | Title Recall | Comm Tokens | Total Tokens | Calls / Q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_context_single` | 0.6800 | 0.8069 | 0.7867 | 0.6532 | 0.8400 | 0.00 | 1984.76 | 1.00 |
| `split_no_comm_mv3` | 0.4000 | 0.5172 | 0.4086 | 0.2574 | 0.3950 | 0.00 | 3001.95 | 3.00 |
| `answer_only_exchange` | 0.4900 | 0.5965 | 0.5306 | 0.3740 | 0.5600 | 58.86 | 6596.12 | 6.00 |
| `evidence_exchange` | 0.6100 | 0.7065 | 0.5816 | 0.4614 | 0.6500 | 364.92 | 6889.65 | 6.00 |
| `full_packet_exchange` | 0.6100 | 0.7051 | 0.5937 | 0.4607 | 0.6400 | 1007.16 | 7503.84 | 6.00 |

## 3. 关键 Delta

| Comparison | Ans EM Δ | Sup F1 Δ | Joint F1 Δ | Comm Tokens Δ |
| --- | ---: | ---: | ---: | ---: |
| `evidence_exchange - split_no_comm_mv3` | 0.2100 | 0.1729 | 0.2040 | 364.92 |
| `full_packet_exchange - answer_only_exchange` | 0.1200 | 0.0630 | 0.0867 | 948.30 |
| `full_context_single - evidence_exchange` | 0.0700 | 0.2051 | 0.1919 | -364.92 |

## 4. 机制与校验摘要

- 题数：`100`
- split 视图数：`300`；full-context 参考视图数：`100`
- 每个方法均导出 HotpotQA 官方预测文件格式：`hotpot_predictions/{method}.json`，包含 `answer` 与 `sp`。
- 报告口径：Answer 使用 EM/F1，Supporting Facts 使用 sentence-level EM/F1，Joint 使用 answer 与 support 的联合指标。

## 5. 分数据集表

### hotpotqa

| Method | Ans EM | Sup F1 | Joint F1 | Comm Tokens | Total Tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `full_context_single` | 0.6800 | 0.7867 | 0.6532 | 0.00 | 1984.76 |
| `split_no_comm_mv3` | 0.4000 | 0.4086 | 0.2574 | 0.00 | 3001.95 |
| `answer_only_exchange` | 0.4900 | 0.5306 | 0.3740 | 58.86 | 6596.12 |
| `evidence_exchange` | 0.6100 | 0.5816 | 0.4614 | 364.92 | 6889.65 |
| `full_packet_exchange` | 0.6100 | 0.5937 | 0.4607 | 1007.16 | 7503.84 |

## 6. 资料来源

- HotpotQA official：https://hotpotqa.github.io/
- HotpotQA GitHub eval format：https://github.com/hotpotqa/hotpot
- HuggingFace HotpotQA dataset card：https://huggingface.co/datasets/hotpotqa/hotpot_qa
- HotpotQA paper：https://arxiv.org/abs/1809.09600
- AgentsNet OpenReview：https://openreview.net/forum?id=gsSIH0mZ0Y

## 7. 局限

- 当前只覆盖 `pilot100` 阶段，不直接作为跨 phase 的显著性结论。
- 本轮优先验证 HotpotQA evidence exchange，AgentsNet 拓扑协调留作后续扩展。
