# `providers` 目录说明

这个目录存放模型 provider 的统一访问封装。

## 当前职责

- 读取 `ResolvedModelConfig`
- 组装兼容 OpenAI Chat Completions 的请求
- 处理超时、重试、用量估算与文本通道抽取

## 维护约定

- 新增 provider 行为时，优先通过配置和映射扩展
- 尽量保持上层 runner 不感知具体 HTTP 细节
