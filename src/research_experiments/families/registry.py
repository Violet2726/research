"""实验家族注册中心。

注册中心只保存家族元数据与惰性导入路径，不在模块导入时直接加载任何 family。
这样共享核心层可以完全不依赖具体实验实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable


RunnerFn = Callable[..., Any]
ValidatorFn = Callable[[str], dict[str, Any]]
SummarizerFn = Callable[[str], dict[str, Any]]
ReportRendererFn = Callable[[str, str | None], dict[str, Any]]
ConfigLoaderFn = Callable[[str], Any]
ModelResolverFn = Callable[[str], Any]
CliMainFn = Callable[[list[str] | None], None]


def _load_object(path: str) -> Any:
    module_path, object_name = path.split(":", 1)
    module = import_module(module_path)
    return getattr(module, object_name)


@dataclass(frozen=True)
class FamilySpec:
    """一个实验家族的统一对外合同。"""

    family_name: str
    config_loader_path: str
    model_resolver_path: str
    runner_path: str
    validator_path: str
    summarizer_path: str
    report_renderer_path: str
    cli_main_path: str

    @property
    def config_loader(self) -> ConfigLoaderFn:
        return _load_object(self.config_loader_path)

    @property
    def model_resolver(self) -> ModelResolverFn:
        return _load_object(self.model_resolver_path)

    @property
    def runner(self) -> RunnerFn:
        return _load_object(self.runner_path)

    @property
    def validator(self) -> ValidatorFn:
        return _load_object(self.validator_path)

    @property
    def summarizer(self) -> SummarizerFn:
        return _load_object(self.summarizer_path)

    @property
    def report_renderer(self) -> ReportRendererFn:
        return _load_object(self.report_renderer_path)

    @property
    def cli_main(self) -> CliMainFn:
        return _load_object(self.cli_main_path)


FAMILY_SPECS: dict[str, FamilySpec] = {
    "budget_comm": FamilySpec(
        family_name="budget_comm",
        config_loader_path="research_experiments.families.budget_comm.config:load_experiment_config",
        model_resolver_path="research_experiments.families.shared.config_loading:resolve_model",
        runner_path="research_experiments.families.budget_comm.run.execute:run_experiment",
        validator_path="research_experiments.families.budget_comm.run.validate:validate_run",
        summarizer_path="research_experiments.families.budget_comm.run.report:summarize_run",
        report_renderer_path="research_experiments.families.budget_comm.run.report:render_report",
        cli_main_path="research_experiments.families.budget_comm.spec:main",
    ),
    "comm_necessary": FamilySpec(
        family_name="comm_necessary",
        config_loader_path="research_experiments.families.comm_necessary.config:load_experiment_config",
        model_resolver_path="research_experiments.families.shared.config_loading:resolve_model",
        runner_path="research_experiments.families.comm_necessary.run.execute:run_experiment",
        validator_path="research_experiments.families.comm_necessary.run.validate:validate_run",
        summarizer_path="research_experiments.families.comm_necessary.run.report:summarize_run",
        report_renderer_path="research_experiments.families.comm_necessary.run.report:render_report",
        cli_main_path="research_experiments.families.comm_necessary.spec:main",
    ),
    "cue": FamilySpec(
        family_name="cue",
        config_loader_path="research_experiments.families.cue.config:load_experiment_config",
        model_resolver_path="research_experiments.families.shared.config_loading:resolve_model",
        runner_path="research_experiments.families.cue.run.execute:run_experiment",
        validator_path="research_experiments.families.cue.run.validate:validate_run",
        summarizer_path="research_experiments.families.cue.run.report:summarize_run",
        report_renderer_path="research_experiments.families.cue.run.report:render_report",
        cli_main_path="research_experiments.families.cue.spec:main",
    ),
    "free_mad_lite": FamilySpec(
        family_name="free_mad_lite",
        config_loader_path="research_experiments.families.free_mad_lite.config:load_experiment_config",
        model_resolver_path="research_experiments.families.shared.config_loading:resolve_model",
        runner_path="research_experiments.families.free_mad_lite.run.execute:run_experiment",
        validator_path="research_experiments.families.free_mad_lite.run.validate:validate_run",
        summarizer_path="research_experiments.families.free_mad_lite.run.report:summarize_run",
        report_renderer_path="research_experiments.families.free_mad_lite.run.report:render_report",
        cli_main_path="research_experiments.families.free_mad_lite.spec:main",
    ),
    "imad": FamilySpec(
        family_name="imad",
        config_loader_path="research_experiments.families.imad.config:load_experiment_config",
        model_resolver_path="research_experiments.families.shared.config_loading:resolve_model",
        runner_path="research_experiments.families.imad.run.execute:run_experiment",
        validator_path="research_experiments.families.imad.run.validate:validate_run",
        summarizer_path="research_experiments.families.imad.run.report:summarize_run",
        report_renderer_path="research_experiments.families.imad.run.report:render_report",
        cli_main_path="research_experiments.families.imad.spec:main",
    ),
    "multi_agent": FamilySpec(
        family_name="multi_agent",
        config_loader_path="research_experiments.families.multi_agent.config:load_experiment_config",
        model_resolver_path="research_experiments.families.shared.config_loading:resolve_model",
        runner_path="research_experiments.families.multi_agent.run.execute:run_experiment",
        validator_path="research_experiments.families.multi_agent.run.validate:validate_run",
        summarizer_path="research_experiments.families.multi_agent.run.report:summarize_run",
        report_renderer_path="research_experiments.families.multi_agent.run.report:render_report",
        cli_main_path="research_experiments.families.multi_agent.spec:main",
    ),
    "selective_comm": FamilySpec(
        family_name="selective_comm",
        config_loader_path="research_experiments.families.selective_comm.config:load_experiment_config",
        model_resolver_path="research_experiments.families.shared.config_loading:resolve_model",
        runner_path="research_experiments.families.selective_comm.run.execute:run_experiment",
        validator_path="research_experiments.families.selective_comm.run.validate:validate_run",
        summarizer_path="research_experiments.families.selective_comm.run.report:summarize_run",
        report_renderer_path="research_experiments.families.selective_comm.run.report:render_report",
        cli_main_path="research_experiments.families.selective_comm.spec:main",
    ),
    "sid_lite": FamilySpec(
        family_name="sid_lite",
        config_loader_path="research_experiments.families.sid_lite.config:load_experiment_config",
        model_resolver_path="research_experiments.families.shared.config_loading:resolve_model",
        runner_path="research_experiments.families.sid_lite.run.execute:run_experiment",
        validator_path="research_experiments.families.sid_lite.run.validate:validate_run",
        summarizer_path="research_experiments.families.sid_lite.run.report:summarize_run",
        report_renderer_path="research_experiments.families.sid_lite.run.report:render_report",
        cli_main_path="research_experiments.families.sid_lite.spec:main",
    ),
    "single_agent": FamilySpec(
        family_name="single_agent",
        config_loader_path="research_experiments.families.single_agent.config:load_experiment_config",
        model_resolver_path="research_experiments.families.shared.config_loading:resolve_model",
        runner_path="research_experiments.families.single_agent.run.execute:run_experiment",
        validator_path="research_experiments.families.single_agent.run.validate:validate_run",
        summarizer_path="research_experiments.families.single_agent.run.report:summarize_run",
        report_renderer_path="research_experiments.families.single_agent.run.report:render_report",
        cli_main_path="research_experiments.families.single_agent.spec:main",
    ),
    "sparc": FamilySpec(
        family_name="sparc",
        config_loader_path="research_experiments.families.sparc.config:load_experiment_config",
        model_resolver_path="research_experiments.families.shared.config_loading:resolve_model",
        runner_path="research_experiments.families.sparc.run.execute:run_experiment",
        validator_path="research_experiments.families.sparc.run.validate:validate_run",
        summarizer_path="research_experiments.families.sparc.run.report:summarize_run",
        report_renderer_path="research_experiments.families.sparc.run.report:render_report",
        cli_main_path="research_experiments.families.sparc.spec:main",
    ),
}


def get_family_spec(family_name: str) -> FamilySpec:
    """按 family 名返回注册合同。"""

    return FAMILY_SPECS[family_name]


def registered_family_names() -> tuple[str, ...]:
    """返回稳定排序后的 family 名列表。"""

    return tuple(sorted(FAMILY_SPECS))


def validator_map() -> dict[str, ValidatorFn]:
    """返回按 family 名索引的校验函数映射。"""

    return {
        family_name: spec.validator
        for family_name, spec in FAMILY_SPECS.items()
    }


