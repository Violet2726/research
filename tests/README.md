# `tests` 目录说明

`tests/` 存放仓库的自动化测试。

## 当前目录结构

- `core/`：共享核心层、benchmark、faithful analysis、matrix 编排测试
- `cli/`：CLI 冒烟测试
- `contracts/`：运行产物与报告合同测试
- `methods/`：按实验线拆分的逻辑与 prompting 测试
- `maintenance/`：导入边界、产物清理等工程约束测试

`conftest.py` 保留在 `tests/` 根目录，供所有子目录共享。

## 常用命令

```powershell
uv run pytest
uv run pytest tests/core/test_research_experiments.py tests/cli/test_cli_smoke.py
uv run pytest tests/methods
```

`__pycache__/` 是解释器缓存目录，不作为人工维护对象。
