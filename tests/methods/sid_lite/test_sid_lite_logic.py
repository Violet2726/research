"""覆盖 SID-lite 机制逻辑与消息包行为的测试。"""

from __future__ import annotations

from sid_lite.logic import decide_early_exit, project_message_packet


def test_sid_lite_high_confidence_consensus_early_exits() -> None:
    rows = [
        {"agent_id": 1, "normalized_answer": "42", "confidence_valid": True, "confidence_value": 0.9},
        {"agent_id": 2, "normalized_answer": "42", "confidence_valid": True, "confidence_value": 0.86},
        {"agent_id": 3, "normalized_answer": "42", "confidence_valid": True, "confidence_value": 0.88},
    ]
    decision = decide_early_exit(rows, mean_conf_threshold=0.75, conf_spread_threshold=0.2)
    assert decision.early_exit is True
    assert decision.triggered is False


def test_sid_lite_invalid_confidence_fails_open() -> None:
    rows = [
        {"agent_id": 1, "normalized_answer": "yes", "confidence_valid": True, "confidence_value": 0.9},
        {"agent_id": 2, "normalized_answer": "yes", "confidence_valid": False, "confidence_value": None},
        {"agent_id": 3, "normalized_answer": "yes", "confidence_valid": True, "confidence_value": 0.91},
    ]
    decision = decide_early_exit(rows, mean_conf_threshold=0.75, conf_spread_threshold=0.2)
    assert decision.early_exit is False
    assert decision.reason == "invalid_confidence_fail_open"


def test_sid_lite_compressed_packet_respects_cap() -> None:
    row = {
        "agent_id": 1,
        "normalized_answer": "yes",
        "confidence_valid": True,
        "confidence_value": 0.81,
        "claim_span": "x" * 500,
        "key_evidence": "y" * 500,
        "uncertain_point": "z" * 500,
    }
    packet = project_message_packet(row, mode="compressed", token_cap=20)
    assert packet["approx_packet_tokens"] <= 20
    assert packet["packet_mode"] == "compressed"
