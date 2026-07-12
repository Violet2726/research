"""RCTA 轨迹综合与预算匹配 MAD 提示。"""

from __future__ import annotations

import json

from research_experiments.core.data.datasets import DatasetSample

RCTA_PROMPT_VERSION = "rcta_trace_synthesis_v1_2"
RCTA_SCHEMA_VERSION = "rcta_trace_certificate_v1_1"


def build_synthesis_messages(sample: DatasetSample, board: str) -> list[dict[str, str]]:
    schema = {
        "reasoning_summary": "concise synthesis, target 80 words and never exceed 120 words",
        "final_answer": "judgeable final answer only; string preferred, JSON scalar allowed",
        "source_trace_ids": ["T1"],
        "decisive_claim": "one decisive check",
        "certificate_type": "arithmetic|symbolic|ordering|boolean|unsupported",
        "certificate_payload": {},
    }
    return [
        {
            "role": "system",
            "content": (
                "You aggregate independent reasoning traces. Re-solve the problem using trace-level complementarity; "
                "do not merely follow the majority. Return exactly one JSON object and no markdown. "
                "Do not expose step-by-step deliberation: emit only the short summary. Keep reasoning_summary under 80 words "
                "and keep the entire JSON compact. Never report confidence. "
                "Only claim a certificate when its payload is mechanically checkable."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{sample.question}\n\nAnonymous traces:\n{board}\n\n"
                "Return this exact JSON schema:\n"
                + json.dumps(schema, ensure_ascii=False)
                + "\nCertificate payloads: arithmetic={expression,claimed_value}; "
                "symbolic={left,right,substitutions}; ordering={items,ordered_items,direction}; "
                "boolean={expression,variables,claimed_value}. Use unsupported with {} when no safe certificate applies."
            ),
        },
    ]


def build_debate_update_messages(sample: DatasetSample, own: dict, peers: list[dict], *, confidence_mode: bool) -> list[dict[str, str]]:
    peer_text = "\n\n".join(
        f"Peer {index}:\n{str((row.get('validated_output') or {}).get('reasoning') or row.get('assistant_text') or '')}\n"
        f"FINAL: {row.get('normalized_answer') or row.get('prediction') or ''}"
        for index, row in enumerate(peers, start=1)
    )
    confidence_instruction = (
        "Communicate a calibrated confidence number from 0 to 1 and condition revisions on the peers' evidence, not rhetoric. "
        if confidence_mode else ""
    )
    return [
        {"role": "system", "content": "You are revising one answer after one controlled multi-agent discussion round."},
        {
            "role": "user",
            "content": (
                f"Question:\n{sample.question}\n\nYour previous response:\n{own.get('assistant_text','')}\n\nOther responses:\n{peer_text}\n\n"
                + confidence_instruction
                + "Return REASONING then FINAL_ANSWER using the standard tagged-line contract."
            ),
        },
    ]
