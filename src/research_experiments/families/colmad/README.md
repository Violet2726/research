# colmad

`colmad` 用于运行协作监督式多智能体辩论论文复现实验。

## 入口

- CLI：`research_cli family colmad`
- 配置：`configs/families/colmad/`
- 默认运行目录：`local/runs/colmad/<experiment>/<phase>/<run_id>/`
- 默认报告目录：`local/reports/colmad/`

## 常用命令

```powershell
uv run research_cli family colmad inspect-experiment --experiment configs/families/colmad/experiments/colmad_realmistake_main.toml
uv run research_cli family colmad run --experiment configs/families/colmad/experiments/colmad_realmistake_main.toml --phase count20 --model xiaomimimo/mimo-v2.5
uv run research_cli family colmad render-report --run-dir local/runs/colmad/colmad_realmistake_main/count20/<run_id>
```

## 当前口径

- `colmad_realmistake_main` 是当前项目里协作监督协议的正式复现主线。
- 这条线进入 `reproduction_matrix`，但不进入 `faithful_matrix`。
- canonical benchmark 固定为 `ReaLMistake` 的三类错误检测任务：
  - `math_problem_generation`
  - `fine_grained_fact_verification`
  - `answerability_classification`
- canonical 方法固定为 `single_agent_detector / copmad_competitive / colmad_collaborative`。

## 论文逻辑对齐

- `single_agent_detector` 提供单智能体错误检测基线。
- `copmad_competitive` 复现竞争式零和辩论，核心风险是 debate hacking、过度自信和说服式误导。
- `colmad_collaborative` 复现支持性批评与证据互补的协作监督协议。
- `judge` 只做最终二值裁决：`contains_error / contains_no_error`。

## 设计边界

- v1 重点是复现 `competitive vs collaborative` 协议差异，而不是把任务改写成一般推理准确率竞赛。
- v1 不额外引入新的 vote/router/verifier，也不混入异质模型实验设计。
- 若后续 `count300` 只体现出微弱增益，或主要依赖明显更高 token 支撑，这条线应冻结为 supporting reproduction。
