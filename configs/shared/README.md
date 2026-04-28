# `configs/shared` 目录说明

这个目录存放跨实验共享的配置资源。

## 当前内容

- `benchmarks/`：benchmark 定义与冻结 split
- `providers/`：provider 默认配置
- `model_catalog.toml`：模型目录、标签与覆盖项

## 维护约定

- 任何实验线都可以引用这里的内容
- 新增共享 benchmark 或 provider 时，优先先补这里
