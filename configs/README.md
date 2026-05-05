# configs

`configs/` 存放整个仓库的实验配置。

## 分层规则

- `shared/`：跨实验共享的 benchmark、provider 和 model catalog
- `<family>/`：单个实验家族自己的 experiment、protocol、policy、method 等配置

## 命名约定

- 配置文件名、`name` 字段和运行目录中的 `<experiment>` 片段保持一致。
- 对外公开的 experiment 名统一使用 `snake_case`。
- 模型切换优先通过 CLI `--model` 或矩阵运行时覆盖，不再为不同模型复制一套 experiment 名。

## 维护约定

- 公共 benchmark / provider / model 信息优先放到 `configs/shared/`。
- experiment 配置负责声明“跑什么”，protocol / policy / method 配置负责声明“怎么跑”。
- 避免在配置目录里保留重复别名或模型专属历史副本。
