from research_experiments.core.config import load_benchmark_config
from research_experiments.core.data.datasets import select_samples
from research_experiments.families.disagreement_guided_crux_reconstruction.config import (
    load_experiment_config,
    load_protocol_config,
)


def test_dgcr_config_is_frozen_and_gate_splits_are_disjoint() -> None:
    experiment = load_experiment_config("configs/families/disagreement_guided_crux_reconstruction/experiments/dgcr_gate.toml")
    protocol = load_protocol_config(experiment.protocol)
    assert protocol.stage_candidates == 5
    assert protocol.resample_candidates == 3
    assert protocol.panel_count == 2
    benchmark = load_benchmark_config("configs/core/shared/benchmarks/bbeh/bbeh-main.toml")
    development = select_samples(benchmark, "dgcr_dev100_seed42")
    heldout = select_samples(benchmark, "dgcr_holdout200_seed42")
    assert len(development) == 100
    assert len(heldout) == 200
    assert not {sample.sample_id for sample in development} & {sample.sample_id for sample in heldout}
    assert len({sample.metadata["task"] for sample in development}) == 23
    assert len({sample.metadata["task"] for sample in heldout}) == 23
    assert sorted(Counter(sample.metadata["task"] for sample in development).values()) == [4] * 15 + [5] * 8
    assert sorted(Counter(sample.metadata["task"] for sample in heldout).values()) == [8] * 7 + [9] * 16
    manifests = [
        json.loads(Path("configs/core/shared/benchmarks/splits/dgcr_dev100/bbeh/bbeh-main-seed42.json").read_text(encoding="utf-8")),
        json.loads(Path("configs/core/shared/benchmarks/splits/dgcr_holdout200/bbeh/bbeh-main-seed42.json").read_text(encoding="utf-8")),
    ]
    assert {manifest["population_sha256"] for manifest in manifests} == {
        "4e93a68bf533b7bbed5daa7038c52ac43cf56c48ec4cddb7315d3f174f9da7ff"
    }
    assert {manifest["sample_ids_sha256"] for manifest in manifests} == {
        "f595491be1c13585f3d25a6e870307e379ec11194a17263006b4db6f9b53f9c5",
        "844fdcdf2dbb7be7ea6f424d9182992da2952c907a97f20e36095c3d445a8b30",
    }
from collections import Counter
import json
from pathlib import Path
