from __future__ import annotations

import hashlib
import json

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.families.contrastive_active_testing.replay import replay_canonicalization


def test_canonicalization_replay_is_zero_network_and_does_not_rewrite_source(tmp_path) -> None:
    run = tmp_path / "historical"
    (run / "turns").mkdir(parents=True)
    (run / "views").mkdir()
    sample = DatasetSample(
        "bbeh",
        "s1",
        "q",
        "B",
        "",
        {"task": "unit", "options": [{"label": "A", "text": "alpha"}, {"label": "B", "text": "beta"}]},
    )
    answers = ["A", "A", "A", "B", "B"]
    rows = []
    for index, answer in enumerate(answers, start=1):
        rows.append(
            {
                "sample_id": "s1",
                "task": "unit",
                "role": "stage_a_solver",
                "agent_id": index,
                "assistant_text": f"REASONING: r{index}\nFINAL_ANSWER: ({answer})",
                "answer_class_key": answer,
                "normalized_answer": answer,
            }
        )
    for index in range(1, 4):
        rows.append(
            {
                "sample_id": "s1",
                "task": "unit",
                "role": "independent_resample",
                "agent_id": index,
                "assistant_text": "REASONING: beta\nFINAL_ANSWER: (B) beta",
                "answer_class_key": "(B) beta",
                "normalized_answer": "(B) beta",
            }
        )
    source = "".join(json.dumps(row) + "\n" for row in rows)
    (run / "turns" / "agent_turns.jsonl").write_text(source, encoding="utf-8")
    (run / "views" / "predictions.jsonl").write_text("{}\n", encoding="utf-8")
    before = hashlib.sha256((run / "turns" / "agent_turns.jsonl").read_bytes()).hexdigest()

    output = tmp_path / "replay.json"
    payload = replay_canonicalization(run, samples=[sample], output_path=output)

    after = hashlib.sha256((run / "turns" / "agent_turns.jsonl").read_bytes()).hexdigest()
    assert before == after
    assert payload["network_requests"] == 0
    assert payload["metrics"]["sc5_micro"] == 0.0
    assert payload["metrics"]["adaptive_sc8_micro"] == 1.0
    assert payload["metrics"]["candidate_oracle_micro"] == 1.0
    assert output.exists()
