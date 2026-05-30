"""约束实验配置里的标准 comparator 口径，防止 matched_controls 再次漂移。"""

from __future__ import annotations

import tomllib
from pathlib import Path

from research_experiments.families.shared.standard_method_names import (
    COT_1,
    MAD_FIXED_R1,
    MV_5,
    MV_6,
    MV_7,
)

ROOT = Path(__file__).resolve().parents[2]


def _load_toml(relative_path: str) -> dict:
    return tomllib.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def test_multi_agent_and_imad_budget_matched_controls_stay_aligned() -> None:
    multi_agent = _load_toml("configs/families/multi_agent/experiments/same_context_controlled_debate.toml")
    setup = multi_agent["setups"][0]
    assert setup["name"] == "mad_3a_r1"
    assert setup["matched_controls"] == [MV_6]

    imad = _load_toml("configs/families/imad/experiments/imad_same_context_main.toml")
    methods = {row["name"]: row for row in imad["methods"]}
    assert methods[MAD_FIXED_R1]["matched_controls"] == [MV_6]
    assert methods["mad_fixed_r2"]["matched_controls"] == []
    assert methods["mad_fixed_r3"]["matched_controls"] == []
    assert methods["imad_adaptive"]["matched_controls"] == []


def test_consensagent_and_madjudge_controls_stay_aligned() -> None:
    consensagent = _load_toml("configs/families/consensagent/experiments/consensagent_main.toml")
    setups = {row["name"]: row for row in consensagent["setups"]}
    assert setups["consensagent_3a"]["matched_controls"] == [COT_1, MV_6]
    assert setups["mad_3a_r1"]["matched_controls"] == [COT_1, MV_6]
    assert setups["mad_3a_r2"]["matched_controls"] == [COT_1, MV_6]

    madjudge = _load_toml("configs/families/madjudge/experiments/madjudge_main.toml")
    judge_setups = {row["name"]: row for row in madjudge["setups"]}
    assert judge_setups["madjudge_7a"]["matched_controls"] == [COT_1, MV_7]
    assert judge_setups["madjudge_5a"]["matched_controls"] == [COT_1, MV_5]
