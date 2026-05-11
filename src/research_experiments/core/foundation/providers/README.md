# research_experiments.core.foundation.providers

这里放共享 provider 适配层。

## 责任

- 统一请求入口
- 统一重试、限流协同与错误归一化
- 统一缓存键依赖字段
- 统一 provider 级软拒答恢复策略

## 维护约定

- 不在实验家族里复制 provider 重试逻辑。
- 新增 provider 兼容修复时，优先在这里做共享实现。
