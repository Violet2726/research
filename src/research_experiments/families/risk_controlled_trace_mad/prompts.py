"""EVF-MAD 的票数盲化选择、对称审计与交叉答辩提示。"""

from __future__ import annotations

from research_experiments.core.data.datasets import DatasetSample

EVF_PROMPT_VERSION = "evf_mad_v4_1"
EVF_SELECTOR_SCHEMA_VERSION = "evf_challenger_selector_v1"
EVF_AUDIT_SCHEMA_VERSION = "evf_symmetric_audit_v1"


def build_selector_messages(sample: DatasetSample, board: str, *, anchor_label: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You select one existing non-anchor candidate for a falsification test. "
                "Vote counts and candidate order are hidden. Do not invent an answer. "
                "Return one JSON object only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{sample.question}\n\nAnonymous candidates:\n{board}\n\n"
                f"The current anchor is Candidate {anchor_label}. Select exactly one different label.\n"
                '{"challenger_label":"A","decisive_difference":"brief falsifiable difference"}'
            ),
        },
    ]


def build_audit_messages(sample: DatasetSample, board: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Act as a symmetric verifier. Evaluate both anonymous candidates independently; do not infer vote counts. "
                "Prefer an existing candidate only when evidence supports it. Emit bounded JSON only. "
                "Every evidence item must bind target_label and use one safe test_type: arithmetic, symbolic, collection, boolean, graph, unsupported. "
                "Use at most 10 evidence items. Support must recompute the target with relation eq; falsification must compare a recomputed value directly against the target with relation ne. "
                "Never output Python code, imports, files, network operations, confidence scores, or a new answer."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{sample.question}\n\nCandidates:\n{board}\n\n"
                "Return this schema:\n"
                '{"preferred_label":"A","decisive_claim":"...","evidence":['
                '{"target_label":"A","claim_kind":"support|falsify",'
                '"test_type":"arithmetic|symbolic|collection|boolean|graph|unsupported","payload":{}}]}\n'
                "For arithmetic/symbolic payload use left,right,relation where relation is exactly eq or ne. "
                "For collection use items,expected_items,mode,relation. "
                "For boolean use expression,variables,expected. "
                "For graph use edges,source,target,reachable,directed."
            ),
        },
    ]


def build_cross_exam_messages(
    sample: DatasetSample,
    board: str,
    *,
    assigned_label: str,
    opposing_claim: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Test the assigned candidate against the opposing audit. You may repair its reasoning or falsify it, "
                "but you may not change candidates. Return the same audit JSON schema with one preferred_label and bounded evidence."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{sample.question}\n\nCandidates:\n{board}\n\n"
                f"Assigned candidate: {assigned_label}\nOpposing claim: {opposing_claim}\n"
                "Return JSON using the symmetric audit schema."
            ),
        },
    ]


# Historical constant aliases are kept inside the sole family only for old artifact readers.
RCTA_PROMPT_VERSION = EVF_PROMPT_VERSION
RCTA_SCHEMA_VERSION = EVF_AUDIT_SCHEMA_VERSION
