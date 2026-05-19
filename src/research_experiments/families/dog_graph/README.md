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
  - 项目内 canonical DoG 复现主线
  - 使用官方 DoG 数据集，并保留论文主流程中的动态关系检索、enough-answer 判断、三角色顺序问题简化与 direct fallback
  - 当前 Freebase 任务默认协议仍是本机可运行口径：`freebase_backend_mode = "local_reduced"`，不是全量 Virtuoso / Freebase 严格后端
- `dog_graph_static_ablation`
  - legacy 静态候选子图消融
  - 只保留当前静态图求解、投票与显式图辩论对照

## 当前实际口径

- `dog_graph` 是平行复现支线，当前进入 `reproduction_matrix`，但不进入 `faithful_matrix`。
- `dog_graph_main` 是项目内唯一 canonical DoG 主线，但当前仍应解释为“流程高保真、本机近似后端”的复现线，不应直接包装成论文数值级严格复现。
- 当前主对照只有两条：`tog_iterative_baseline` 与 `dog_graph_paper`。
- `tog_iterative_baseline` 是项目内结构消融基线：保留动态检索与逐跳回答，但不使用三角色顺序问题简化；它不是官方仓中原始 ToG 的一比一实现。
- Freebase 任务当前默认走 `local_reduced` 局部 KG 后端；MetaQA 任务走 `metaqa/kb.txt` 轻量图后端。
- 如果需要更接近论文原始实验口径，Freebase 任务仍应优先切回本地 Virtuoso / SPARQL 后端。

## 当前结果判断

- 截至 `count300 / 20260515T033459Z-xiaomimimo-mimo-v2.5`，`dog_graph_paper` 相比 `tog_iterative_baseline` 的总体准确率增益约为 `+0.0171`，但平均总 token 增加到 `4.4149x`。
- 关键任务上，`WebQuestions` 为 `+0.0167`，`GrailQA` 为 `+0.0200`，方向与论文一致，但提升幅度显著小于论文摘要中强调的优势。
- `MetaQA 2-hop` 仍有正增益，但 `MetaQA 3-hop` 出现 `-0.0167` 的负增益，说明当前口径下 DoG 机制还没有稳定覆盖所有任务。
- 因此，这条线目前更适合解释为“主流程跑通、趋势部分成立”，还不适合宣称“已经复现出论文同等级的显著优势”。

## 为什么增益小但 token 高

- `dog_graph_paper` 比基线多出的主要开销来自三角色顺序问题简化；每个未解决 hop 都会增加额外调用，而且简化 prompt 会携带问题、triples、历史对话与角色说明。
- 当前 `local_reduced` Freebase 后端削弱了论文中 DoG 最该发挥价值的那部分“关系筛选与路径重整”收益，因此新增 token 没有转化成同等级的准确率提升。
- 项目内 `tog_iterative_baseline` 本身已经保留了动态检索与逐跳回答，所以 DoG 当前是在一个并不弱的基线上做结构增量，而不是在极简基线上获得巨幅提升。

## 后续建议

- 若目标是追近论文数值幅度，优先事项不是继续微调 prompt，而是为 Freebase 任务恢复更接近论文设定的 Virtuoso / SPARQL 后端。
- 若目标是提高“论文一致性”，下一步应补一个更贴近官方仓的 ToG 严格对照，而不只保留当前 `tog_iterative_baseline` 结构消融线。
- 在恢复严格后端前，建议把当前 `dog_graph_main` 结果定位为 supporting reproduction，不要把 `count300` 的小幅增益包装成论文级主结论。
- 如果继续沿本机 `local_reduced` 路线迭代，更值得优先优化的是成本控制，例如减少不必要的三角色简化调用，而不是单纯追求更多 token。
