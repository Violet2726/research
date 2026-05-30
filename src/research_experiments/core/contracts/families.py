"""实验家族平台合同模型。"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

FamilyPrototype = Literal[
    "independent_sampling",
    "debate_rounds",
    "shared_stage_policy",
    "packet_belief_update",
    "topology_or_graph",
]


ArtifactPaths = dict[str, Path | tuple[Path, ...]]


class FamilyArtifactContract(BaseModel):
    """声明某个 family 对外暴露的正式运行产物合同。"""

    model_config = ConfigDict(frozen=True)

    manifest_path: str = "manifest.json"
    progress_path: str = "progress.json"
    validation_path: str = "run_validation.json"
    report_path: str = "report.md"
    figure_manifest_path: str = "figure_manifest.json"
    metrics_view_path: str
    prediction_records_path: str
    turn_record_paths: tuple[str, ...] = ()
    extra_view_paths: tuple[str, ...] = ()


class FamilyManifest(BaseModel):
    """单个实验家族的统一平台注册合同。"""

    model_config = ConfigDict(frozen=True)

    family_name: str
    prototype: FamilyPrototype
    historical_labels: tuple[str, ...] = ()
    config_loader_path: str
    model_resolver_path: str
    runner_path: str
    validator_path: str
    summarizer_path: str
    report_renderer_path: str
    cli_main_path: str
    artifact_contract: FamilyArtifactContract

    def _load_object(self, path: str) -> Callable[..., object]:
        module_path, object_name = path.split(":", 1)
        module = import_module(module_path)
        loaded = getattr(module, object_name)
        if not callable(loaded):
            raise TypeError(f"{path} 不是可调用对象。")
        return loaded

    @property
    def config_loader(self) -> Callable[..., object]:
        return self._load_object(self.config_loader_path)

    @property
    def model_resolver(self) -> Callable[..., object]:
        return self._load_object(self.model_resolver_path)

    @property
    def runner(self) -> Callable[..., object]:
        return self._load_object(self.runner_path)

    @property
    def validator(self) -> Callable[..., object]:
        return self._load_object(self.validator_path)

    @property
    def summarizer(self) -> Callable[..., object]:
        return self._load_object(self.summarizer_path)

    @property
    def report_renderer(self) -> Callable[..., object]:
        return self._load_object(self.report_renderer_path)

    @property
    def cli_main(self) -> Callable[..., object]:
        return self._load_object(self.cli_main_path)

    def build_artifact_paths(self, run_dir: str | Path) -> ArtifactPaths:
        """把相对合同路径映射成某个 run 目录下的绝对路径。"""

        root = Path(run_dir)
        contract = self.artifact_contract
        return {
            "manifest_path": root / contract.manifest_path,
            "progress_path": root / contract.progress_path,
            "validation_path": root / contract.validation_path,
            "report_path": root / contract.report_path,
            "figure_manifest_path": root / contract.figure_manifest_path,
            "metrics_view_path": root / contract.metrics_view_path,
            "prediction_records_path": root / contract.prediction_records_path,
            "turn_record_paths": tuple(root / path for path in contract.turn_record_paths),
            "extra_view_paths": tuple(root / path for path in contract.extra_view_paths),
        }
