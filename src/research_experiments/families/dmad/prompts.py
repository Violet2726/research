"""DMAD family 的提示词构造。"""

from __future__ import annotations

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.core.prompts.dataset_contracts import build_json_system_prompt, dataset_instruction_for_sample
from research_experiments.families.dmad.config import AgentProfile
from research_experiments.families.shared.reasoning_methods import resolve_reasoning_method


DEFAULT_PROMPT_VERSION = "dmad_v1_json"


def build_reasoning_stage_messages(
    sample: DatasetSample,
    agent_profile: AgentProfile,
    *,
    round_index: int,
    prior_rounds: list[list[dict[str, str]]] | None = None,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    _assert_prompt_version(prompt_version)
    method_spec = resolve_reasoning_method(sample.dataset, agent_profile.strategy_name)
    history_block = _render_prior_rounds(agent_profile.agent_id, prior_rounds or [])
    user_prompt = (
        f"{_agent_header(agent_profile, method_spec)}"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    if history_block:
        user_prompt += f"Debate history so far:\n{history_block}\n\n"
    user_prompt += (
        f"You are now in round {round_index} of the debate.\n"
        "Write only the solving process for this round.\n"
        "Keep using your assigned reasoning method.\n"
        "First audit your own previous solution, then compare it against peer solutions.\n"
        "Change your path only when you can name one concrete flaw in your previous attempt and one concrete reason the revision is better supported.\n"
        f"When peer solutions disagree, {method_spec.debate_action}.\n"
        "Do not output JSON and do not output a separate final-answer line.\n"
    )
    if sample.dataset in {"mmlu_pro", "gpqa_diamond", "mmlu_abstract_algebra"}:
        user_prompt += "Keep the solving process under 180 words and focus only on the decisive option-by-option checks.\n"
    if method_spec.label == "PoT":
        user_prompt += (
            "Output only executable Python code for the decisive computation.\n"
            "Do not explain the code and do not add markdown fences.\n"
            'The program must store the final computed result in a variable named "ans".\n'
            "If you revise the code, repair the concrete bug directly in the program.\n"
        )
    counting_hint = _counting_probability_guardrails(sample)
    if counting_hint:
        user_prompt += f"\nMath verification hint: {counting_hint}"
    guardrails = _multiple_choice_guardrails(sample)
    if guardrails:
        user_prompt += f"\nFor answer-format safety: {guardrails}"
    return [
        {"role": "system", "content": _pot_process_system_prompt() if method_spec.label == "PoT" else _plain_text_system_prompt()},
        {"role": "user", "content": user_prompt},
    ]


def build_answer_stage_messages(
    sample: DatasetSample,
    agent_profile: AgentProfile,
    *,
    round_index: int,
    solving_process: str,
    execution_result: str | None = None,
    execution_status: str | None = None,
    execution_error: str | None = None,
    prior_rounds: list[list[dict[str, str]]] | None = None,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    _assert_prompt_version(prompt_version)
    method_spec = resolve_reasoning_method(sample.dataset, agent_profile.strategy_name)
    history_block = _render_prior_rounds(agent_profile.agent_id, prior_rounds or [])
    user_prompt = (
        f"You are agent_{agent_profile.agent_id} in round {round_index} of a Diverse Multi-Agent Debate experiment.\n"
        f"Reasoning method: {method_spec.label}\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    if history_block:
        user_prompt += f"Debate history so far:\n{history_block}\n\n"
    user_prompt += f"Your current solving process:\n{solving_process.strip()}\n\n"
    if method_spec.label == "PoT":
        if execution_status == "ok" and execution_result:
            user_prompt += (
                f'Program execution result from variable "ans": {execution_result}\n'
                "Treat this execution result as authoritative unless you can point to a concrete code bug.\n\n"
            )
        elif execution_result:
            user_prompt += (
                f'Program execution result recovered from the code: {execution_result}\n'
                f"Program execution note: {_execution_note(execution_status)}\n"
                "Treat this recovered result as authoritative unless you can point to a concrete code bug.\n\n"
            )
        elif execution_status and execution_status != "ok":
            user_prompt += (
                f"Program execution note: {_execution_note(execution_status)}\n"
                f"Program execution detail: {_execution_detail(execution_error)}\n"
                "If the code failed, repair the answer only when you can identify the concrete bug.\n\n"
            )
    user_prompt += (
        "Return exactly one JSON object with key \"final_answer\".\n"
        "Do not include a reasoning field.\n"
        "Do not add any extra text.\n"
    )
    counting_hint = _counting_probability_guardrails(sample)
    if counting_hint:
        user_prompt += f"\nMath verification hint: {counting_hint}"
    guardrails = _multiple_choice_guardrails(sample)
    if guardrails:
        user_prompt += f"\nFor answer-format safety: {guardrails}"
    return [
        {"role": "system", "content": _answer_only_system_prompt()},
        {"role": "user", "content": user_prompt},
    ]


def build_initial_messages(
    sample: DatasetSample,
    agent_profile: AgentProfile,
    *,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    _assert_prompt_version(prompt_version)
    method_spec = resolve_reasoning_method(sample.dataset, agent_profile.strategy_name)
    user_prompt = (
        f"{_agent_header(agent_profile, method_spec)}"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    if method_spec.label == "PoT":
        user_prompt += (
            "Solve independently with your assigned reasoning method.\n"
            "Return exactly one JSON object with keys \"final_answer\", \"reasoning\", and \"python_program\".\n"
            'The field "python_program" must be the shortest executable Python code, without comments or markdown fences, and it must store the final computed result in a variable named "ans".\n'
            "Keep the reasoning concise but inspection-ready.\n"
            "Make final_answer consistent with the program result.\n"
            "Do not add any extra text.\n"
        )
    else:
        user_prompt += (
            "Solve independently with your assigned reasoning method.\n"
            "Return exactly one JSON object with keys \"final_answer\" and \"reasoning\".\n"
            "Keep the reasoning concise but inspection-ready.\n"
            "Do not add any extra text.\n"
        )
    if sample.dataset in {"mmlu_pro", "gpqa_diamond", "mmlu_abstract_algebra"}:
        user_prompt += "Keep the reasoning under 180 words and focus only on the decisive option-by-option checks.\n"
    counting_hint = _counting_probability_guardrails(sample)
    if counting_hint:
        user_prompt += f"\nMath verification hint: {counting_hint}"
    guardrails = _multiple_choice_guardrails(sample)
    if guardrails:
        user_prompt += f"\nFor answer-format safety: {guardrails}"
    return [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": user_prompt},
    ]


def build_reflection_feedback_messages(
    sample: DatasetSample,
    previous_reasoning: str,
    previous_answer: str,
    *,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    _assert_prompt_version(prompt_version)
    user_prompt = (
        "You are the critique stage of a reflective single-agent baseline.\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += (
        f"Previous reasoning:\n{previous_reasoning}\n\n"
        f"Previous final_answer: {previous_answer}\n\n"
        "Review the draft once.\n"
        "Name the most concrete mistake, missing evidence, or unsupported jump if one exists.\n"
        "If the draft already looks sound, say that it is supported and name the strongest supporting check.\n"
        "Return plain text only and keep it under 120 words.\n"
    )
    return [
        {"role": "system", "content": _reflection_feedback_system_prompt()},
        {"role": "user", "content": user_prompt},
    ]


def build_reflection_revision_messages(
    sample: DatasetSample,
    previous_reasoning: str,
    previous_answer: str,
    feedback: str,
    *,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    _assert_prompt_version(prompt_version)
    user_prompt = (
        "You are the revise stage of a reflective single-agent baseline.\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += (
        f"Previous reasoning:\n{previous_reasoning}\n\n"
        f"Previous final_answer: {previous_answer}\n\n"
        f"Reviewer feedback:\n{feedback.strip()}\n\n"
        "Revise only if the feedback identifies a concrete issue.\n"
        "Return exactly one JSON object with keys \"final_answer\" and \"reasoning\".\n"
        "Keep the reasoning focused on the decisive correction or confirmation.\n"
    )
    guardrails = _multiple_choice_guardrails(sample)
    if guardrails:
        user_prompt += f"\nFor answer-format safety: {guardrails}"
    return [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": user_prompt},
    ]


def build_self_contrast_checklist_messages(
    sample: DatasetSample,
    candidate_solutions: list[dict[str, str]],
    *,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    _assert_prompt_version(prompt_version)
    candidates_block = _render_candidate_solutions(candidate_solutions)
    user_prompt = (
        "You are constructing a contrastive checklist for a Self-Contrast baseline.\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += (
        f"Candidate solutions:\n{candidates_block}\n\n"
        "Compare the candidates.\n"
        "List the concrete disagreements, likely failure points, and the checks a final solver must satisfy.\n"
        "Return plain text only.\n"
    )
    return [
        {"role": "system", "content": _plain_text_system_prompt()},
        {"role": "user", "content": user_prompt},
    ]


def build_self_contrast_revision_messages(
    sample: DatasetSample,
    candidate_solutions: list[dict[str, str]],
    checklist: str,
    *,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    _assert_prompt_version(prompt_version)
    candidates_block = _render_candidate_solutions(candidate_solutions)
    user_prompt = (
        "You are the final revision stage of a Self-Contrast baseline.\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += (
        f"Candidate solutions:\n{candidates_block}\n\n"
        f"Contrastive checklist:\n{checklist.strip()}\n\n"
        "Use the checklist to synthesize the strongest supported answer.\n"
        "Return exactly one JSON object with keys \"final_answer\" and \"reasoning\".\n"
        "Do not add any extra text.\n"
    )
    guardrails = _multiple_choice_guardrails(sample)
    if guardrails:
        user_prompt += f"\nFor answer-format safety: {guardrails}"
    return [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": user_prompt},
    ]


def build_mrp_method_selection_messages(
    sample: DatasetSample,
    candidate_methods: list[str],
    *,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    _assert_prompt_version(prompt_version)
    rendered_methods = []
    for method_name in candidate_methods:
        method_spec = resolve_reasoning_method(sample.dataset, method_name)
        rendered_methods.append(f"- {normalize_reasoning_method_label(sample.dataset, method_name)}: {method_spec.summary}")
    user_prompt = (
        "You are the routing stage of a Meta-Reasoning Prompting baseline.\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += (
        "Candidate reasoning methods:\n"
        + "\n".join(rendered_methods)
        + "\n\nChoose the single most suitable reasoning method for this question.\n"
        "Return exactly one JSON object with keys \"selected_method\" and \"reasoning\".\n"
        "selected_method must be one of the provided method names in lowercase.\n"
    )
    return [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": user_prompt},
    ]


def build_mrp_solution_messages(
    sample: DatasetSample,
    method_name: str,
    *,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    _assert_prompt_version(prompt_version)
    agent_profile = AgentProfile(
        agent_id=1,
        persona_name="general_reasoner",
        persona_instruction="Act as a careful general-purpose reasoning assistant.",
        strategy_name=method_name,
        strategy_instruction="",
    )
    return build_initial_messages(sample, agent_profile, prompt_version=prompt_version)


def normalize_reasoning_method_label(dataset: str, method_name: str) -> str:
    return resolve_reasoning_method(dataset, method_name).label.lower()


def _agent_header(agent_profile: AgentProfile, method_spec) -> str:
    return (
        f"You are agent_{agent_profile.agent_id} in a Diverse Multi-Agent Debate experiment.\n"
        f"Persona: {agent_profile.persona_name}\n"
        f"Persona guidance: {agent_profile.persona_instruction}\n"
        f"Reasoning method: {method_spec.label}\n"
        f"Method summary: {method_spec.summary}\n"
        f"Method guidance: {method_spec.guidance}\n"
        f"Method checklist: {method_spec.checklist}\n"
    )


def _render_candidate_solutions(candidate_solutions: list[dict[str, str]]) -> str:
    return "\n\n".join(
        "\n".join(
            [
                f"Candidate {index + 1}:",
                f"- method: {item['strategy_name']}",
                f"- reasoning: {item['reasoning'].strip()}",
                f"- final_answer: {item['answer'].strip()}",
            ]
        )
        for index, item in enumerate(candidate_solutions)
    )


def _system_prompt() -> str:
    return build_json_system_prompt(
        "You are one reasoning agent in a controlled DMAD reproduction experiment.",
        extra_rules=[
            "Solve the task carefully using only the provided question and context.",
            "Respect the assigned reasoning method.",
            "Keep the reasoning field concise but substantive.",
            "Do not add text before or after the JSON object.",
        ],
    )


def _plain_text_system_prompt() -> str:
    return "\n".join(
        [
            "You are one reasoning agent in a controlled DMAD reproduction experiment.",
            "Return plain text only.",
            "Do not use markdown fences.",
            "Solve the task carefully using only the provided context.",
        ]
    )


def _pot_process_system_prompt() -> str:
    return "\n".join(
        [
            "You are one reasoning agent in a controlled DMAD reproduction experiment.",
            "Return plain Python code only.",
            "Do not explain the code.",
            "Do not use markdown fences.",
            'The code must store the final computed result in a variable named "ans".',
        ]
    )


def _answer_only_system_prompt() -> str:
    return build_json_system_prompt(
        "You are one reasoning agent in a controlled DMAD reproduction experiment.",
        extra_rules=[
            'Return exactly one JSON object with key "final_answer".',
            "Do not add a reasoning field unless explicitly requested.",
            "Do not add text before or after the JSON object.",
        ],
    )


def _reflection_feedback_system_prompt() -> str:
    return "\n".join(
        [
            "You are a concise critique stage for a reflective single-agent baseline.",
            "Return plain text only.",
            "Name the decisive flaw or the decisive confirmation.",
            "Do not output JSON.",
        ]
    )


def _dataset_instruction(sample: DatasetSample) -> str:
    if sample.dataset == "hotpotqa":
        return dataset_instruction_for_sample(sample, hotpot_style="shortest_span_copy")
    return dataset_instruction_for_sample(sample)


def _assert_prompt_version(prompt_version: str) -> None:
    if prompt_version != DEFAULT_PROMPT_VERSION:
        raise ValueError(f"Unsupported DMAD prompt_version: {prompt_version}")


def _multiple_choice_guardrails(sample: DatasetSample) -> str:
    if sample.dataset not in {"mmlu_pro", "gpqa_diamond", "mmlu_abstract_algebra"}:
        return ""
    return (
        "do not switch options just because two peers agree; "
        "switch only when you can name one concrete flaw in the previous option and one concrete reason the new option is better supported; "
        "if a minority answer is the only one backed by explicit calculation or option-by-option elimination, prefer the better-supported minority; "
        "before returning, compare your option against each distinct peer option and keep it only if it wins on explicit evidence; "
        "if you compute a content answer, map it back to the matching option letter before returning; "
        "if the task depends on structure, stereochemistry, or named positions, verify each named feature against the chosen option before returning; "
        'end the reasoning by explicitly stating "Best option: <LETTER>" and make final_answer match that letter exactly'
    )


def _counting_probability_guardrails(sample: DatasetSample) -> str:
    if sample.dataset != "competition_math":
        return ""
    subject = str(sample.metadata.get("subject") or "").strip().lower()
    if subject not in {"counting_and_probability", "counting & probability"}:
        return ""
    return (
        "for counting or probability problems, verify the result with at least one of: a complement count, a disjoint case split, or a reverse sanity check; if letters repeat or objects are indistinguishable, explicitly check for overcounting; do not use Cartesian-product-with-replacement counting unless replacement is explicitly allowed by the problem"
    )


def _execution_note(status: str | None) -> str:
    mapping = {
        "missing_program": "the previous code block was incomplete",
        "missing_result": "the previous code did not expose a final result",
        "runtime_error": "the previous code raised a runtime issue",
        "unsafe_program": "the previous code used unsupported constructs",
    }
    return mapping.get(str(status or "").strip(), "the previous code did not complete successfully")


def _execution_detail(error: str | None) -> str:
    cleaned = str(error or "").strip()
    if not cleaned:
        return "unknown"
    return cleaned[:180]


def _render_prior_rounds(agent_id: int, prior_rounds: list[list[dict[str, str]]]) -> str:
    if not prior_rounds:
        return ""
    blocks: list[str] = []
    for round_rows in prior_rounds:
        if not round_rows:
            continue
        round_number = round_rows[0].get("round_index", "?")
        own_rows = [row for row in round_rows if int(row.get("agent_id") or -1) == agent_id]
        peer_rows = [row for row in round_rows if int(row.get("agent_id") or -1) != agent_id]
        blocks.append(f"Round {round_number}:")
        if own_rows:
            own = own_rows[0]
            own_lines = [
                "Your previous record:",
                f"- method: {own.get('strategy_name', '')}",
                f"- solving process: {str(own.get('reasoning') or '').strip()}",
                f"- final answer: {str(own.get('answer') or '').strip()}",
            ]
            if str(own.get("execution_result") or "").strip():
                own_lines.append(f"- execution result: {str(own.get('execution_result') or '').strip()}")
            blocks.append("\n".join(own_lines))
        if peer_rows:
            peer_lines = ["Other agents' records:"]
            for peer in peer_rows:
                peer_lines.extend(
                    [
                        f"- agent_{peer.get('agent_id')}: method={peer.get('strategy_name', '')}",
                        f"  solving process: {str(peer.get('reasoning') or '').strip()}",
                        f"  final answer: {str(peer.get('answer') or '').strip()}",
                    ]
                )
                if str(peer.get("execution_result") or "").strip():
                    peer_lines.append(f"  execution result: {str(peer.get('execution_result') or '').strip()}")
            blocks.append("\n".join(peer_lines))
    return "\n\n".join(blocks)
