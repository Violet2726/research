# HotpotQA 通信必要性 Smoke20 报告

## 1. 实验概览

- 实验名：`hotpotqa_split_evidence_v1`
- Phase：`smoke20`
- Backbone：`xiaomimimo/mimo-v2.5`
- 运行目录：`runs/comm_necessary/hotpotqa_split_evidence_v1/smoke20/20260505T050231Z-hotpotqa_split_evidence_v1-smoke20-xiaomimimo-mimo-v2.5`
- 任务：HotpotQA split-context evidence exchange；smoke20 只作工程验证和方向性证据。
- 方法：`full_context_single`、`split_no_comm_mv3`、`answer_only_exchange`、`evidence_exchange`、`full_packet_exchange`。

## 2. 主结果表

| Method | Ans EM | Ans F1 | Sup F1 | Joint F1 | Title Recall | Comm Tokens | Total Tokens | Calls / Q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_context_single` | 0.6000 | 0.6829 | 0.7288 | 0.5239 | 0.8000 | 0.00 | 1947.10 | 1.00 |
| `split_no_comm_mv3` | 0.2500 | 0.3692 | 0.5312 | 0.1976 | 0.5250 | 0.00 | 2965.15 | 3.00 |
| `answer_only_exchange` | 0.4000 | 0.5192 | 0.5772 | 0.3266 | 0.6500 | 78.70 | 6535.85 | 6.00 |
| `evidence_exchange` | 0.4000 | 0.5192 | 0.5629 | 0.3115 | 0.6000 | 426.50 | 6858.45 | 6.00 |
| `full_packet_exchange` | 0.4500 | 0.5692 | 0.5611 | 0.3792 | 0.6500 | 1076.50 | 7502.65 | 6.00 |

## 3. 关键 Delta

| Comparison | Ans EM Δ | Sup F1 Δ | Joint F1 Δ | Comm Tokens Δ |
| --- | ---: | ---: | ---: | ---: |
| `evidence_exchange - split_no_comm_mv3` | 0.1500 | 0.0317 | 0.1139 | 426.50 |
| `full_packet_exchange - answer_only_exchange` | 0.0500 | -0.0162 | 0.0526 | 997.80 |
| `full_context_single - evidence_exchange` | 0.2000 | 0.1660 | 0.2124 | -426.50 |

## 4. 机制与校验摘要

- 题数：`20`
- split 视图数：`60`；full-context 参考视图数：`20`
- 每个方法均导出 HotpotQA 官方预测文件格式：`hotpot_predictions/{method}.json`，包含 `answer` 与 `sp`。
- 报告口径：Answer 使用 EM/F1；Supporting Facts 使用 sentence-level EM/F1；Joint 使用 answer 与 support 的联合指标。

## 5. 分数据集表

### hotpotqa

| Method | Ans EM | Sup F1 | Joint F1 | Comm Tokens | Total Tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `full_context_single` | 0.6000 | 0.7288 | 0.5239 | 0.00 | 1947.10 |
| `split_no_comm_mv3` | 0.2500 | 0.5312 | 0.1976 | 0.00 | 2965.15 |
| `answer_only_exchange` | 0.4000 | 0.5772 | 0.3266 | 78.70 | 6535.85 |
| `evidence_exchange` | 0.4000 | 0.5629 | 0.3115 | 426.50 | 6858.45 |
| `full_packet_exchange` | 0.4500 | 0.5611 | 0.3792 | 1076.50 | 7502.65 |

## 6. 资料来源

- HotpotQA official：https://hotpotqa.github.io/
- HotpotQA GitHub eval format：https://github.com/hotpotqa/hotpot
- HuggingFace HotpotQA dataset card：https://huggingface.co/datasets/hotpotqa/hotpot_qa
- HotpotQA paper：https://arxiv.org/abs/1809.09600
- AgentsNet OpenReview：https://openreview.net/forum?id=gsSIH0mZ0Y

## 7. 局限

- 当前只运行 smoke20，不作为全量显著性结论。
- 本轮优先验证 HotpotQA evidence exchange，AgentsNet 拓扑协调留作后续扩展。
