# CRED-V / CRED-CVS

`cred_v` 当前以 `cred_rfs_repair_only_v6` 为稳定回退，以 CRED-CVS
（Certificate-Verified Search）为前向研究线。所有前向方法共享 5 路自由 CoT
Stage A；模型只能提出候选与证书，本地 checker 决定是否晋级。

主线方法：

- `cred_rfs_vote_5_anchor`：与 `sc_5` 对齐的共享投票锚点。
- `cred_rfs_repair_only_v6`：仅执行 scorer-safe / context-supported 确定性修复。
- `cred_cvs_budget_matched_vote_v1`：使用相同额外调用但忽略证书的预算对照。
- `cred_cvs_v1`：两个独立模型提出同一候选，题面绑定证书均通过后才允许晋级。
- `cred_isp_shadow_v1`：二阶信念聚合影子实验，不改变最终答案。

旧 verifier、pairwise selector、semantic promotion 与 v3/v4/v5/v7/v8/v9
仅保留在显式 `legacy_experiment = true` 的复现配置中，不进入默认主线。

开发实验：

```bash
uv run research_cli experiment --family cred_v inspect-experiment --experiment configs/families/cred_v/experiments/cred_v_cvs_v1.toml
uv run research_cli experiment --family cred_v run --experiment configs/families/cred_v/experiments/cred_v_cvs_v1.toml --phase count20
uv run research_cli experiment --family cred_v run --experiment configs/families/cred_v/experiments/cred_v_cvs_v1.toml --phase count100
uv run research_cli experiment --family cred_v run --experiment configs/families/cred_v/experiments/cred_v_cvs_v1.toml --phase count300
```

确认性实验使用 `cred_v_cvs_locked.toml`；跨基准实验使用
`cred_v_cvs_transfer.toml`。后者的 `pilot` 与 `locked` 采用分层窗口，样本 ID
互不重叠。每次 run 完成后执行统一校验与报告：

```bash
uv run research_cli experiment --family cred_v validate-run --run-dir local/runs/cred_v/cred_v_cvs_v1/count20/<run_id>
uv run research_cli experiment --family cred_v render-report --run-dir local/runs/cred_v/cred_v_cvs_v1/count20/<run_id>
```
