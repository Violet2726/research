"""平台层公开记录模型。"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class PredictionRecord(BaseModel):
    """统一的题级预测记录。"""

    model_config = ConfigDict(extra="allow", frozen=True)

    dataset: str | None = None
    sample_id: str | None = None
    method_name: str | None = None
    score: float | None = None


class TurnRecord(BaseModel):
    """统一的调用级 turn 记录。"""

    model_config = ConfigDict(extra="allow", frozen=True)

    dataset: str | None = None
    sample_id: str | None = None
    method_name: str | None = None
    role: str | None = None
    output_status: str | None = None


class MatrixEntryRecord(BaseModel):
    """矩阵级语义条目的规范读模型。"""

    model_config = ConfigDict(extra="allow", frozen=True)

    family: str
    experiment_name: str
    config_path: str
    status: str
    run_dir: str | None = None


class ExperimentSpec(BaseModel):
    """实验配置在平台层的最小规范视图。"""

    model_config = ConfigDict(extra="allow", frozen=True)

    name: str
    primary_model_ref: str
    phase_name: str | None = None
    description: str | None = None


class ReportInputModel(BaseModel):
    """报告层读取 run 产物时使用的规范输入视图。"""

    model_config = ConfigDict(extra="allow", frozen=True)

    family_name: str
    run_dir: Path
    metrics_view_path: Path
    prediction_records_path: Path
    report_path: Path
    figure_manifest_path: Path


__all__ = [
    "ExperimentSpec",
    "MatrixEntryRecord",
    "PredictionRecord",
    "ReportInputModel",
    "TurnRecord",
]
