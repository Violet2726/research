"""共享数据加载、split 与评分入口。"""

from __future__ import annotations

from research_experiments.core.data.datasets import (
    DatasetSample,
    generate_split_manifests,
    load_samples,
    load_split_ids,
    resolve_dataset_source_path,
    resolve_split_manifest_path,
    select_samples,
)
from research_experiments.core.data.evaluation import (
    aggregate_majority,
    normalize_gold,
    normalize_math_expression,
    normalize_multiple_choice,
    normalize_number,
    normalize_prediction,
    normalize_text,
    normalize_yes_no,
    score_multiple_choice,
    score_prediction,
)

__all__ = [
    "DatasetSample",
    "load_samples",
    "generate_split_manifests",
    "load_split_ids",
    "select_samples",
    "resolve_split_manifest_path",
    "resolve_dataset_source_path",
    "normalize_prediction",
    "normalize_gold",
    "score_prediction",
    "aggregate_majority",
    "normalize_number",
    "normalize_yes_no",
    "normalize_text",
    "normalize_multiple_choice",
    "normalize_math_expression",
    "score_multiple_choice",
]
