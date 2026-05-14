"""DoG 图推理复现的提示词构造器。"""

from __future__ import annotations

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.core.prompts.dataset_contracts import build_json_system_prompt, dataset_instruction_for_sample
from research_experiments.families.dog_graph.dataset_views import GraphView


DEFAULT_PROMPT_VERSION = "dog_graph_json_v1"


def build_initial_messages(
    sample: DatasetSample,
    graph_view: GraphView,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    """构造首轮独立图求解消息。"""

    user_prompt = (
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
        f"Visible graph view ({graph_view.view_kind}):\n{graph_view.context_text}\n\n"
        f"Role guidance:\n{_role_guidance(graph_view.view_kind)}\n\n"
        f"{_json_contract()}\n"
        "Use only triples or path fragments that appear either in the visible graph view or in the global graph snapshot shown inside it. "
        "Do not invent graph evidence. "
        "Prefer canonical entity titles over paraphrases, abbreviations, or shortened surface forms. "
        "Return JSON only."
    )
    return [
        {"role": "system", "content": _system_prompt(prompt_version)},
        {"role": "user", "content": user_prompt},
    ]


def build_debate_messages(
    sample: DatasetSample,
    graph_view: GraphView,
    round_index: int,
    previous_answer: str,
    previous_reasoning: str,
    previous_evidence_triples: list[str],
    previous_answer_path: list[str],
    peer_messages: list[dict[str, object]],
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    """构造图辩论轮的消息。"""

    peer_block = "\n\n".join(
        [
            f"{item['agent']} answer: {item['answer']}\n"
            f"{item['agent']} reasoning: {item['reasoning']}\n"
            f"{item['agent']} evidence_triples: {item['evidence_triples']}\n"
            f"{item['agent']} answer_path: {item['answer_path']}"
            for item in peer_messages
        ]
    ) or "No peer feedback."
    user_prompt = (
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
        f"Visible graph view ({graph_view.view_kind}):\n{graph_view.context_text}\n\n"
        f"Role guidance:\n{_role_guidance(graph_view.view_kind)}\n\n"
        f"Debate round: {round_index}\n"
        f"Your previous answer: {previous_answer}\n"
        f"Your previous reasoning: {previous_reasoning}\n"
        f"Your previous evidence_triples: {previous_evidence_triples}\n"
        f"Your previous answer_path: {previous_answer_path}\n\n"
        f"Peer feedback:\n{peer_block}\n\n"
        f"{_json_contract()}\n"
        "Compare peer candidate answers against your own graph evidence. "
        "You may adopt peer evidence if it is quoted explicitly in peer feedback and is consistent with the graph snapshot you can see. "
        "Only revise your answer if peer evidence exposes a concrete graph mismatch, a better entity disambiguation, or a more canonical title. "
        "If you keep or change the answer, cite exact triples or path fragments from your visible graph snapshot or exact peer-cited triples. "
        "Prefer canonical entity titles such as full person names, full organization names, official currency names, and language titles. "
        "Return JSON only."
    )
    return [
        {"role": "system", "content": _system_prompt(prompt_version)},
        {"role": "user", "content": user_prompt},
    ]


def _system_prompt(prompt_version: str) -> str:
    if prompt_version != DEFAULT_PROMPT_VERSION:
        raise ValueError(f"Unsupported DoG prompt_version: {prompt_version}")
    return build_json_system_prompt(
        "You are one graph-reasoning agent in a Debate on Graph reproduction experiment.",
        extra_rules=[
            "Use only the provided graph evidence and the question.",
            "Keep reasoning short and evidence-grounded.",
            "Every evidence_triples item must be copied or lightly normalized from the visible graph view.",
            "Do not add markdown fences or prose outside the JSON object.",
        ],
    )


def _json_contract() -> str:
    return (
        'Return exactly one JSON object with keys "final_answer", "reasoning", '
        '"evidence_triples", and "answer_path". '
        '"final_answer" must be the shortest judgeable answer span. '
        '"reasoning" must stay under 70 words. '
        '"evidence_triples" must be an array of 1-4 explicit triples or path fragments. '
        '"answer_path" must be an array of 1-4 ordered path steps.'
    )


def _dataset_instruction(sample: DatasetSample) -> str:
    return dataset_instruction_for_sample(sample, context_scope="visible", hotpot_style="shortest_span")


def _role_guidance(view_kind: str) -> str:
    if view_kind == "relation_path_view":
        return (
            "- Trace the relation path from topic seed to answer slot.\n"
            "- When multiple answers are plausible, choose the one whose title most naturally satisfies the relation.\n"
            "- For languages, currencies, and roles, prefer full canonical titles rather than shortened mentions."
        )
    if view_kind == "entity_neighborhood_view":
        return (
            "- Use linked entities, clue phrases, and nearby nodes to disambiguate named entities.\n"
            "- If the question implies a current spouse, official currency, or official language, prefer the current or official entity title.\n"
            "- Avoid generic placeholders such as 'answer', 'unknown', or bare country names when a canonical title is inferable."
        )
    if view_kind == "evidence_summary_view":
        return (
            "- Synthesize the highest-support triples, concept candidates, and query sketch.\n"
            "- Prefer the most canonical answer title that is consistent with both the graph and the question wording.\n"
            "- If peers disagree later, break ties in favor of the answer with the strongest explicit graph evidence."
        )
    return "- Use the graph evidence conservatively and prefer canonical answer titles."
