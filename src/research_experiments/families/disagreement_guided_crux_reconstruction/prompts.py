"""冻结的 DGCR 提示；绝不渲染候选支持数或推理轨迹。"""

from __future__ import annotations

from research_experiments.core.data.datasets import DatasetSample


DGCR_PROMPT_VERSION = "dgcr_v1"


def build_proposer_messages(sample: DatasetSample, *, label_to_key: dict[str, str]) -> list[dict[str, str]]:
    candidates = "\n".join(f"- Candidate {label}: {key}" for label, key in label_to_key.items())
    return [
        {
            "role": "system",
            "content": (
                "Return JSON only. You select one exact, contiguous source span that could discriminate "
                "the anonymous answer hypotheses. Do not infer vote counts or write a solution."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{sample.question}\n\nAnonymous answer classes:\n{candidates}\n\n"
                "Return exactly {\"start_char\": <integer>, \"end_char\": <integer>}. "
                "The span must be in the question body, not in a trailing Options block, and contain 8-256 characters."
            ),
        },
    ]


def build_panel_messages(
    sample: DatasetSample,
    *,
    masked_question: str,
    label_to_key: dict[str, str],
) -> list[dict[str, str]]:
    candidates = "\n".join(f"- Candidate {label}: {key}" for label, key in label_to_key.items())
    labels = ", ".join(f'\"{label}\"' for label in label_to_key)
    return [
        {
            "role": "system",
            "content": (
                "Return JSON only. For every anonymous candidate hypothesis, reconstruct the exact hidden source "
                "text that would make that candidate true. Do not choose an answer and do not introduce new candidates."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Masked question:\n{masked_question}\n\nAnonymous answer classes:\n{candidates}\n\n"
                "Return exactly {\"reconstructions\": {" + ", ".join(f"{label}: <text>" for label in label_to_key) + "}}. "
                f"The reconstruction keys must be exactly: {labels}."
            ),
        },
    ]
