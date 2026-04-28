# `configs` 目录说明

`configs/` 存放整个仓库的实验配置。

## 分层规则

- `shared/`：跨实验共享的 benchmark、provider、model catalog
- 各实验子目录：只放本实验线自己的 `experiments / protocols / policies / methods / views` 等配置

## 维护约定

- 共享配置优先放到 `configs/shared/`
- 文件命名尽量与论文表格、实验阶段或机制版本对应
- 配置文件统一使用 TOML
