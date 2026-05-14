# dog_graph

`dog_graph` 用于运行 DoG 图推理复现实验。

## 入口

- CLI：`research_cli family dog_graph`
- 配置：`configs/families/dog_graph/`
- 默认运行目录：`local/runs/dog_graph/<experiment>/<phase>/<run_id>/`
- 默认报告目录：`local/reports/dog_graph/`

## 常用命令

```powershell
uv run research_cli family dog_graph inspect-experiment --experiment configs/families/dog_graph/experiments/dog_graph_main.toml
uv run research_cli family dog_graph validate-backend --experiment configs/families/dog_graph/experiments/dog_graph_main.toml
uv run research_cli family dog_graph run --experiment configs/families/dog_graph/experiments/dog_graph_main.toml --phase count20 --model xiaomimimo/mimo-v2.5
uv run research_cli family dog_graph render-report --run-dir local/runs/dog_graph/dog_graph_main/count20/<run_id>
```

## 实验线

- `dog_graph_main`
  - canonical 论文主线
  - 使用官方 DoG 数据集、动态关系检索、enough-answer 判断、三角色顺序问题简化与 direct fallback
- `dog_graph_static_ablation`
  - legacy 静态候选子图消融
  - 只保留当前静态图求解、投票与显式图辩论对照

## 结果定位

- `dog_graph` 是平行复现支线，不直接并入当前 `faithful_matrix`。
- `dog_graph_main` 是唯一 canonical DoG 论文复现实验，不再把静态子图消融包装成主结果。
- Freebase 任务需要本地 Virtuoso / SPARQL 后端；MetaQA 任务需要 `dog-metaqa/kb.txt` 图文件。
