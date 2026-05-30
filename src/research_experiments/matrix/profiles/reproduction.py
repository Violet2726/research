"""reproduction matrix 后处理入口。"""

from research_experiments.matrix.reproduction_analysis import render_reproduction_analysis
from research_experiments.reporting.reproduction_landscape import render_reproduction_landscape
from research_experiments.reporting.reproduction_package import render_reproduction_package


def run_postprocess(state_root, *, reference_state_path_or_root=None) -> None:
    """执行 reproduction 矩阵后处理链路。"""

    del reference_state_path_or_root
    render_reproduction_analysis(state_root)
    render_reproduction_package(state_root)
    render_reproduction_landscape(state_root)
