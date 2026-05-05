# configs/sid_lite

SID-lite 实验配置根目录。

## 目录组成

- `experiments/`：实验入口配置
- `protocols/`：Stage A、压缩通信与 belief update 协议

## 维护约定

- 方法列表、退出规则和 token 上限通过配置控制。
- 不在 experiment 文件里复制共享 benchmark / provider 细节。
