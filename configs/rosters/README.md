# Roster 预留目录

这里预留给未来的多智能体角色配置使用。

计划中的典型角色映射包括：

- `solver -> provider/model`
- `verifier -> provider/model`
- `router -> provider/model`
- `auditor -> provider/model`

当前 baseline runner 还不会读取这个目录；也就是说，现阶段单模型实验
仍然只通过 CLI 的 `--model provider/model` 指定 backbone 模型。
