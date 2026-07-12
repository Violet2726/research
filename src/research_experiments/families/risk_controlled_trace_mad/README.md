# MAD Innovation / EVF-MAD

本目录是唯一活跃的 MAD 创新实验族。BRD、SGSA、RCTA 已在版本注册表中标为淘汰版本；旧代码不再作为独立 family 维护，历史结果仍保留在 `local/runs`，精确复现使用注册表记录的 Git commit。

当前 `v4_evf` 使用固定三路 Qwen-Flash、两路 MiMo-v2.5。出现分歧时，只有异构审计一致、challenger 具有两个可执行通过证据、锚点具有至少一个可执行反例且 challenger 无执行失败证据，才允许覆盖多数答案。

权威说明和命令见 `docs/mad_innovation_mainline.md`。所有阶段只使用 `count*_seed42` 或 `full*_seed42` 命名。

```powershell
uv run research_cli experiment --family risk_controlled_trace_mad inspect-experiment --experiment configs/families/risk_controlled_trace_mad/experiments/mad_innovation.toml
uv run research_cli experiment --family risk_controlled_trace_mad render-report --run-dir local/runs/risk_controlled_trace_mad/mad_innovation/count20_seed42/<run-id>
```
