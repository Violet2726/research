# 多智能体实验配置目录

这里专门放独立的多智能体实验线配置，不和现有 `configs/experiments/` 混用。

目录职责：

- `protocols/`
  debate 协议定义，例如 Vanilla MAD 的轮数与拓扑。
- `rosters/`
  agent 编组模板，例如同构 2-agent、3-agent。
- `controls/`
  debate-vs-vote 使用的等预算单模型对照方法。
- `experiments/`
  具体实验规格，负责把 benchmark、phase、setup 和控制方法组装起来。

使用约定：

- 多智能体实验始终通过 `--backbone provider/model` 显式指定 backbone。
- Vanilla MAD 单次运行只允许一个 backbone，不混异构模型。
- 后续新增 DALA、SID、iMAD 等方法时，优先新增 protocol，再扩 experiment。
