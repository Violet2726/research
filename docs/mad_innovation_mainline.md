# MAD 创新统一主线

`risk_controlled_trace_mad` 是唯一仍可运行的 MAD 创新 family。版本状态以
`configs/families/risk_controlled_trace_mad/versions.toml` 为权威来源。

## 历史结论

- `v1_brd`：淘汰的负结果。强制证伪评审未形成稳定净纠正。
- `v2_sgsa`：淘汰的负结果。一致晋级覆盖不足，未优于 SC 锚点。
- `v3_rcta`：前提失败。MiMo `count100_seed42` 中 GSA 为 7 次纠正、20 次伤害，证书仅 3/200 通过；运行未启用正式 router。
- `v4_evf`：当前版本。以固定 Qwen-Flash + MiMo-v2.5 异构编组和可执行反证控制覆盖伤害。

旧版本精确复现依赖版本注册表中的 Git commit；历史 `local/runs` 不移动、不改写。

## 规范命令

```powershell
uv run research_cli experiment --family risk_controlled_trace_mad inspect-versions --experiment configs/families/risk_controlled_trace_mad/experiments/mad_innovation.toml
uv run research_cli experiment --family risk_controlled_trace_mad inspect-experiment --experiment configs/families/risk_controlled_trace_mad/experiments/mad_innovation.toml
uv run research_cli experiment --family risk_controlled_trace_mad run --experiment configs/families/risk_controlled_trace_mad/experiments/mad_innovation.toml --phase count20_seed42 --version v4_evf
uv run research_cli experiment --family risk_controlled_trace_mad run --experiment configs/families/risk_controlled_trace_mad/experiments/mad_innovation.toml --phase count100_seed42 --version v4_evf
uv run research_cli experiment --family risk_controlled_trace_mad run --experiment configs/families/risk_controlled_trace_mad/experiments/mad_innovation.toml --phase count300_seed42 --version v4_evf
uv run research_cli experiment --family risk_controlled_trace_mad run --experiment configs/families/risk_controlled_trace_mad/experiments/mad_innovation.toml --phase full_seed42 --version v4_evf
```

`count20_seed42` 只用于工程验证。后续阶段由 runner 自动检查前一阶段的 validation 和 progression gate。
