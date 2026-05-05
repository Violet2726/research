# HotpotQA 通信必要性 Smoke20 报告

## 1. 实验概览

- 实验名：`hotpotqa_split_evidence_v1`
- Phase：`smoke20`
- Backbone：`xiaomimimo/mimo-v2.5`
- 运行目录：`runs/comm_necessary/hotpotqa_split_evidence_v1/smoke20/20260505T090449Z-hotpotqa_split_evidence_v1-smoke20-xiaomimimo-mimo-v2.5`
- 任务：HotpotQA split-context evidence exchange；smoke20 只作工程验证和方向性证据。
- 方法：`full_context_single`、`split_no_comm_mv3`、`answer_only_exchange`、`evidence_exchange`、`full_packet_exchange`。

## 2. 主结果表

| Method | Ans EM | Ans F1 | Sup F1 | Joint F1 | Title Recall | Comm Tokens | Total Tokens | Calls / Q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_context_single` | 0.6000 | 0.7012 | 0.7302 | 0.5848 | 0.8250 | 0.00 | 1958.65 | 1.00 |
| `split_no_comm_mv3` | 0.2000 | 0.2650 | 0.4208 | 0.1375 | 0.4500 | 0.00 | 3014.90 | 3.00 |
| `answer_only_exchange` | 0.4500 | 0.5150 | 0.5408 | 0.3058 | 0.5750 | 77.10 | 6562.20 | 6.00 |
| `evidence_exchange` | 0.5500 | 0.6150 | 0.5891 | 0.4530 | 0.6000 | 424.00 | 6912.10 | 6.00 |
| `full_packet_exchange` | 0.5500 | 0.6150 | 0.6319 | 0.4669 | 0.6500 | 1069.20 | 7539.35 | 6.00 |

## 3. 关键 Delta

| Comparison | Ans EM Δ | Sup F1 Δ | Joint F1 Δ | Comm Tokens Δ |
| --- | ---: | ---: | ---: | ---: |
| `evidence_exchange - split_no_comm_mv3` | 0.3500 | 0.1683 | 0.3156 | 424.00 |
| `full_packet_exchange - answer_only_exchange` | 0.1000 | 0.0911 | 0.1611 | 992.10 |
| `full_context_single - evidence_exchange` | 0.0500 | 0.1411 | 0.1317 | -424.00 |

## 4. 机制与校验摘要

- 题数：`20`
- split 视图数：`60`；full-context 参考视图数：`20`
- 每个方法均导出 HotpotQA 官方预测文件格式：`hotpot_predictions/{method}.json`，包含 `answer` 与 `sp`。
- 报告口径：Answer 使用 EM/F1；Supporting Facts 使用 sentence-level EM/F1；Joint 使用 answer 与 support 的联合指标。

## 5. 分数据集表

### hotpotqa

| Method | Ans EM | Sup F1 | Joint F1 | Comm Tokens | Total Tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `full_context_single` | 0.6000 | 0.7302 | 0.5848 | 0.00 | 1958.65 |
| `split_no_comm_mv3` | 0.2000 | 0.4208 | 0.1375 | 0.00 | 3014.90 |
| `answer_only_exchange` | 0.4500 | 0.5408 | 0.3058 | 77.10 | 6562.20 |
| `evidence_exchange` | 0.5500 | 0.5891 | 0.4530 | 424.00 | 6912.10 |
| `full_packet_exchange` | 0.5500 | 0.6319 | 0.4669 | 1069.20 | 7539.35 |

## 6. 资料来源

- HotpotQA official：https://hotpotqa.github.io/
- HotpotQA GitHub eval format：https://github.com/hotpotqa/hotpot
- HuggingFace HotpotQA dataset card：https://huggingface.co/datasets/hotpotqa/hotpot_qa
- HotpotQA paper：https://arxiv.org/abs/1809.09600
- AgentsNet OpenReview：https://openreview.net/forum?id=gsSIH0mZ0Y

## 7. 局限

- 当前只运行 smoke20，不作为全量显著性结论。
- 本轮优先验证 HotpotQA evidence exchange，AgentsNet 拓扑协调留作后续扩展。
