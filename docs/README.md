# docs

`docs/` 存放仓库级设计说明、结构约定和长期维护文档。

## 当前文档

- `project_structure.md`：项目分层、目录职责、默认工作区与 UTF-8 约定
- `run_report_pipeline.md`：实验执行到图资产、正式报告与论文包的完整链路
- `huggingface_archive_workflow.md`：`runs/cache` 的 Hugging Face 归档工作流
- `code_annotation_guidelines.md`：中文注解、docstring 与仓库文档写作规范

## 使用约定

- 跨实验、跨目录的长期规则写在这里
- 单次实验报告不要放在 `docs/`，统一放到 `local/reports/` 或正式归档 run 内
- 临时分析笔记放到 `files/` 或用户自有目录，不回流到 `docs/`
