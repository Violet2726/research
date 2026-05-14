# configs/dog_graph

DoG 图推理复现实验配置根目录。

## 实验入口

- `dog_graph_main`
  - 原论文高保真主实验
  - 使用官方数据集与动态图检索后端
- `dog_graph_static_ablation`
  - legacy 静态候选子图消融
  - 仅作为补充对照

## 说明

- `dog_graph` 是平行研究支线，不并入当前 `faithful_matrix`。
- canonical 主线默认以官方论文与官方仓为 source of truth。
- 若缺少本地 Freebase/Virtuoso 或 MetaQA 图后端，请先运行：

```powershell
uv run research_cli family dog_graph validate-backend --experiment configs/families/dog_graph/experiments/dog_graph_main.toml
```
