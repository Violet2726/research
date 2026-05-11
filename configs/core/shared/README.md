# configs/shared

`configs/core/shared/` 存放跨实验复用的共享配置。

## 当前内容

- `benchmarks/`：benchmark 定义与 frozen split 配置
- `providers/`：provider 默认配置
- `model_catalog.toml`：模型目录、标签和覆盖项

## 维护约定

- 新 benchmark、provider 或模型标签优先补到这里。
- 共享配置一旦被多个实验家族引用，就不要再复制到家族目录里。
