# configs/table_critic

Table-Critic 表推理复现实验配置根目录。

## 实验入口

- `table_critic_main`
  - 原论文主复现实验
  - 使用 `WikiTQ / TabFact` 两套结构化表推理测试文件

## 说明

- `table_critic` 是平行论文复现支线，不并入当前 `faithful_matrix`。
- 当前 v1 只复现 Table-Critic 的核心 `Judge / Critic / Refiner / Curator` 逻辑，不纳入 `Binder / Dater`。
- 若需要先恢复数据资产，请运行：

```powershell
uv run research_cli tools dataset-assets prepare-used
```

