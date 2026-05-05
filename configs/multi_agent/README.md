# configs/multi_agent

多智能体 debate / vote 实验配置根目录。

## 目录组成

- `experiments/`：实验入口配置
- `protocols/`：多轮通信协议
- `rosters/`：agent roster 定义
- `controls/`：matched control 方法

## 维护约定

- `experiments/` 只负责装配协议、roster 和 controls，不在入口文件里展开所有细节。
- minimal / local-only 配置可以保留，但默认不进入全量矩阵。
