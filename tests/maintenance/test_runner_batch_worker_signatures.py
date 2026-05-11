"""约束样本级批处理 worker 的统一签名。"""

from __future__ import annotations

import inspect

from budget_comm.runner import _run_sample as budget_run_sample
from comm_necessary.runner import _run_sample as comm_necessary_run_sample
from cue.runner import _run_sample as cue_run_sample
from free_mad_lite.runner import _run_sample as free_mad_run_sample
from multi_agent.runner import _run_mad_sample as multi_agent_run_sample
from selective_comm.runner import _run_sample as selective_run_sample
from sid_lite.runner import _run_sample as sid_run_sample
from sparc.runner import _run_sample as sparc_run_sample


def test_sample_batch_workers_accept_sample_as_first_positional_argument() -> None:
    worker_functions = [
        budget_run_sample,
        comm_necessary_run_sample,
        cue_run_sample,
        free_mad_run_sample,
        multi_agent_run_sample,
        selective_run_sample,
        sid_run_sample,
        sparc_run_sample,
    ]

    for worker in worker_functions:
        parameters = list(inspect.signature(worker).parameters.values())
        assert parameters, f"{worker.__module__}.{worker.__name__} should define parameters"
        assert parameters[0].name == "sample"
        assert parameters[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
