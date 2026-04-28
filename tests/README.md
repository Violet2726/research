# `tests` 目录说明

`tests/` 存放仓库的自动化测试。

## 当前测试类型

- `test_experiment_core.py`：共享核心层测试
- `test_cli_smoke.py`：CLI 冒烟测试
- `test_run_contracts.py`：运行产物契约测试
- `test_*_logic.py`：各实验线逻辑测试
- `test_import_boundaries.py`：导入边界测试

## 常用命令

```powershell
uv run pytest
uv run pytest tests/test_experiment_core.py tests/test_cli_smoke.py
```

`__pycache__/` 是解释器缓存目录，不作为人工维护对象。
