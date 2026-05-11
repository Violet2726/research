# configs/free_mad_lite

Free-MAD-lite 实验配置根目录。

## 目录组成

- `experiments/`：实验入口配置
- `protocols/`：单轮 debate、trajectory judging 和 fallback 协议

## 维护约定

- anti-conformity 与 trajectory judging 的差异放在 protocol 层表达。
- experiment 层只负责选择协议、phase 和 benchmark。
