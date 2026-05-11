from __future__ import annotations

import io
import json
import sys

from research_experiments.core.foundation.cli_output import emit_json


def test_emit_json_reconfigures_non_utf8_stdout(monkeypatch) -> None:
    raw_stdout = io.BytesIO()
    raw_stderr = io.BytesIO()
    stdout = io.TextIOWrapper(raw_stdout, encoding="cp1252", write_through=True)
    stderr = io.TextIOWrapper(raw_stderr, encoding="cp1252", write_through=True)
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)

    emit_json({"message": "中文输出"})
    sys.stdout.flush()

    payload = json.loads(raw_stdout.getvalue().decode("utf-8"))
    assert payload["message"] == "中文输出"
