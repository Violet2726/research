# `tests` 目录说明

`tests/` 存放仓库的自动化测试。

## 当前目录结构

- `core/`：仅保留共享核心层测试，并按 `config/`、`data/`、`execution/`、`controls/`、`prompts/`、`structured_outputs/` 继续拆分
- `cli/`：CLI 冒烟测试
- `cli_support/`：CLI 输出与 UTF-8 终端编码支持测试
- `workspace/`：工作区布局、归档、HF 同步与数据集资产测试
- `reporting/`：报告、图资产与论文包测试
- `matrix/`：faithful matrix 编排、分析与验收测试
- `families/`：按实验家族拆分的逻辑、prompting 与 family 共享层测试
- `integration/`：运行产物合同与跨层集成合同测试
- `governance/`：导入边界、目录治理与工程约束测试
- `testsupport/`：测试层共享 helper，不放业务断言

`conftest.py` 保留在 `tests/` 根目录，供所有子目录共享。

## 常用命令

```powershell
uv run pytest
uv run pytest tests/core tests/workspace
uv run pytest tests/families tests/integration
uv run pytest tests/governance tests/cli/test_cli_smoke.py
```

`__pycache__/` 是解释器缓存目录，不作为人工维护对象。
