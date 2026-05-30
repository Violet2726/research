"""faithful matrix 后处理入口。"""

from research_experiments.matrix.faithful_acceptance import render_acceptance_summary
from research_experiments.matrix.faithful_analysis import render_faithful_analysis
from research_experiments.reporting.family_landscape import render_family_landscape
from research_experiments.reporting.paper_package import render_paper_package
from research_experiments.reporting.paper_statistics import render_paper_statistics


def run_postprocess(state_root, *, reference_state_path_or_root=None) -> None:
    """执行 faithful 矩阵后处理链路。"""

    render_faithful_analysis(
        state_root,
        reference_state_path_or_root=reference_state_path_or_root,
    )
    render_acceptance_summary(state_root)
    render_paper_statistics(state_root)
    render_paper_package(state_root)
    render_family_landscape(state_root)
