# configs/imad

iMAD 自适应停止实验配置根目录。

## 目录组成

- `experiments/`：正式实验入口配置
- `protocols/`：多轮 debate 与稳定性检测协议

## 当前正式实验

- `imad_same_context_main`

## 维护约定

- `imad` 只承接 same-context 自适应停止复现，不与 split-context 协议混放。
- 固定轮数方法与 adaptive 方法统一在 experiment 层显式声明，避免语义漂移。

