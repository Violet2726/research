# `configs/shared/providers` 目录说明

这个目录存放 provider 的默认连接配置。

## 当前内容

- 各 provider 的 `base_url`
- API Key 环境变量名
- 默认采样参数、超时与重试配置

## 维护约定

- provider 行为差异优先通过配置表达
- 敏感信息不要写进这些文件，统一走环境变量
