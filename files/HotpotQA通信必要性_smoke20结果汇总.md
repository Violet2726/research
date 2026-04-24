# HotpotQA 通信必要性 Smoke20 报告

## 1. 实验概览

- 实验名：`hotpotqa_split_evidence_v1`
- Phase：`smoke20`
- Backbone：`dashscope/qwen-turbo-1101`
- 运行目录：`local/runs/comm_necessary/hotpotqa_split_evidence_v1/smoke20/20260424T083720Z-hotpotqa_split_evidence_v1-smoke20-dashscope-qwen-turbo-1101`
- 任务：HotpotQA split-context evidence exchange；smoke20 只作工程验证和方向性证据。
- 方法：`full_context_single`、`split_no_comm_mv3`、`answer_only_exchange`、`evidence_exchange`、`full_packet_exchange`。

## 2. 主结果表

| Method | Ans EM | Ans F1 | Sup F1 | Joint F1 | Title Recall | Comm Tokens | Total Tokens | Calls / Q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_context_single` | 0.5500 | 0.6542 | 0.4917 | 0.3235 | 0.6000 | 0.00 | 1886.90 | 1.00 |
| `split_no_comm_mv3` | 0.2500 | 0.3150 | 0.3275 | 0.1726 | 0.3750 | 0.00 | 2686.55 | 3.00 |
| `answer_only_exchange` | 0.3000 | 0.3859 | 0.4401 | 0.1960 | 0.6000 | 85.20 | 5751.05 | 6.00 |
| `evidence_exchange` | 0.4000 | 0.4859 | 0.3992 | 0.2383 | 0.5750 | 350.20 | 6021.90 | 6.00 |
| `full_packet_exchange` | 0.4000 | 0.4859 | 0.3906 | 0.2560 | 0.5500 | 602.50 | 6264.60 | 6.00 |

## 3. 关键 Delta

| Comparison | Ans EM Δ | Sup F1 Δ | Joint F1 Δ | Comm Tokens Δ |
| --- | ---: | ---: | ---: | ---: |
| `evidence_exchange - split_no_comm_mv3` | 0.1500 | 0.0717 | 0.0656 | 350.20 |
| `full_packet_exchange - answer_only_exchange` | 0.1000 | -0.0496 | 0.0600 | 517.30 |
| `full_context_single - evidence_exchange` | 0.1500 | 0.0925 | 0.0853 | -350.20 |

## 4. 机制与校验摘要

- 题数：`20`
- split 视图数：`60`；full-context 参考视图数：`20`
- 每个方法均导出 HotpotQA 官方预测文件格式：`hotpot_predictions/{method}.json`，包含 `answer` 与 `sp`。
- 报告口径：Answer 使用 EM/F1；Supporting Facts 使用 sentence-level EM/F1；Joint 使用 answer 与 support 的联合指标。

## 5. 分数据集表

### hotpotqa

| Method | Ans EM | Sup F1 | Joint F1 | Comm Tokens | Total Tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `full_context_single` | 0.5500 | 0.4917 | 0.3235 | 0.00 | 1886.90 |
| `split_no_comm_mv3` | 0.2500 | 0.3275 | 0.1726 | 0.00 | 2686.55 |
| `answer_only_exchange` | 0.3000 | 0.4401 | 0.1960 | 85.20 | 5751.05 |
| `evidence_exchange` | 0.4000 | 0.3992 | 0.2383 | 350.20 | 6021.90 |
| `full_packet_exchange` | 0.4000 | 0.3906 | 0.2560 | 602.50 | 6264.60 |

## 6. 资料来源

- HotpotQA official：https://hotpotqa.github.io/
- HotpotQA GitHub eval format：https://github.com/hotpotqa/hotpot
- HuggingFace HotpotQA dataset card：https://huggingface.co/datasets/hotpotqa/hotpot_qa
- HotpotQA paper：https://arxiv.org/abs/1809.09600
- AgentsNet OpenReview：https://openreview.net/forum?id=gsSIH0mZ0Y

## 7. 局限

- 当前只运行 smoke20，不作为全量显著性结论。
- 本轮优先验证 HotpotQA evidence exchange，AgentsNet 拓扑协调留作后续扩展。
