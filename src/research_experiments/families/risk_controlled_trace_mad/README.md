# MAD Innovation / H-SGSA v5

这是唯一活跃的 MAD 创新实验族。`v5_hsgsa` 固定使用同一个 MiMo-v2.5、同一基础提示和五个确定性种子；`v4_evf` 因零覆盖被标为历史前提失败，不再是可运行主线。更早的 BRD、SGSA、RCTA 仅保留版本注册与历史工件。

## 冻结机制

- Stage-A 共享五次同质独立求解，投票、触发和评分共用保守的无标签答案类键。
- 仅在答案类分歧时追加三次独立重采样和三次支持度盲化审阅；每位审阅者获得独立标签置换及哈希固定的单条代表轨迹。
- H-SGSA 只有在三名有效审阅者一致选择同一现存、非 SC5 锚点答案类时才覆盖；审阅者生成的新答案只作为 shadow 工件。
- CoT1、SC3、SC5、adaptive-SC8、conditional-resample3、单审、2/3 和 3/3 H-SGSA 都从同一物理调用图导出。

## 数据与止损

- `replay_dev_seed42` 是只读阶段，指向历史 BBEH 前 300 题，runner 明确禁止在线调用。
- `confirm_seed42` 在发请求前排除这 300 题，保留 BBEH 4,220 题（23 个任务）和 GPQA Diamond 198 题；最坏 48,598 个逻辑调用。
- 全局真实网络尝试硬停止于 50,000；运行期间不生成中期方法准确率。
- 当前历史回放因审阅输出可用率 95.93% 未达到 99.5% 门槛而失败，因此确认实验不得启动。审计见 `configs/families/risk_controlled_trace_mad/retired/v5_hsgsa_dev_replay_audit.json`。

成功时允许的最强表述仅限：固定 MiMo-v2.5、固定提示与匹配测试时预算下，H-SGSA 在持出 BBEH 上形成新的准确率—成本 Pareto 点。
