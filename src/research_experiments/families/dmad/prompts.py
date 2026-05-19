"""Prompt builders for the DMAD family."""

from __future__ import annotations

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.core.prompts.dataset_contracts import build_json_system_prompt, dataset_instruction_for_sample
from research_experiments.families.dmad.config import AgentProfile


DEFAULT_PROMPT_VERSION = "dmad_v1_json"


def build_reasoning_stage_messages(
    sample: DatasetSample,
    agent_profile: AgentProfile,
    *,
    round_index: int,
    prior_rounds: list[list[dict[str, str]]] | None = None,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    """Build the paper-aligned solving-process prompt for one agent and one round."""

    _assert_prompt_version(prompt_version)
    strategy = _strategy_spec(sample, agent_profile.strategy_name)
    history_block = _render_prior_rounds(agent_profile.agent_id, prior_rounds or [])
    guardrails = _multiple_choice_debate_guardrails(sample)
    user_prompt = (
        f"You are agent_{agent_profile.agent_id} in round {round_index} of a Diverse Multi-Agent Debate experiment.\n"
        f"Persona: {agent_profile.persona_name}\n"
        f"Persona guidance: {agent_profile.persona_instruction}\n"
        f"Reasoning method: {strategy['label']}\n"
        f"Method summary: {strategy['summary']}\n"
        f"Method guidance: {strategy['guidance']}\n"
        f"Method checklist: {strategy['checklist']}\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    if history_block:
        user_prompt += f"Debate history so far:\n{history_block}\n\n"
    user_prompt += (
        "Produce only your solving process for this round. "
        "Keep using your assigned reasoning method. "
        "If you revise, explain only the decisive corrected path rather than restarting from scratch multiple times. "
        "Do not output JSON and do not output the final answer line separately."
    )
    if guardrails:
        user_prompt += f"\nFor answer-format safety: {guardrails}"
    return [
        {"role": "system", "content": _plain_text_system_prompt()},
        {"role": "user", "content": user_prompt},
    ]


def build_answer_stage_messages(
    sample: DatasetSample,
    agent_profile: AgentProfile,
    *,
    round_index: int,
    solving_process: str,
    prior_rounds: list[list[dict[str, str]]] | None = None,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    """Build the paper-aligned final-answer extraction prompt for one agent and one round."""

    _assert_prompt_version(prompt_version)
    strategy = _strategy_spec(sample, agent_profile.strategy_name)
    history_block = _render_prior_rounds(agent_profile.agent_id, prior_rounds or [])
    guardrails = _multiple_choice_debate_guardrails(sample)
    user_prompt = (
        f"You are agent_{agent_profile.agent_id} in round {round_index} of a Diverse Multi-Agent Debate experiment.\n"
        f"Reasoning method: {strategy['label']}\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    if history_block:
        user_prompt += f"Debate history so far:\n{history_block}\n\n"
    user_prompt += (
        f"Your current solving process:\n{solving_process.strip()}\n\n"
        "Based only on the solving process above, return the final answer. "
        'Return exactly one JSON object with key "final_answer". '
        "Do not include a long reasoning field. Return JSON only."
    )
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
    """Build the initial independent-solving prompt."""

    _assert_prompt_version(prompt_version)
    strategy = _strategy_spec(sample, agent_profile.strategy_name)
    guardrails = _multiple_choice_debate_guardrails(sample)
    user_prompt = (
        f"You are agent_{agent_profile.agent_id} in a Diverse Multi-Agent Debate experiment.\n"
        f"Persona: {agent_profile.persona_name}\n"
        f"Persona guidance: {agent_profile.persona_instruction}\n"
        f"Reasoning method: {strategy['label']}\n"
        f"Method summary: {strategy['summary']}\n"
    )
    user_prompt += f"Method guidance: {strategy['guidance']}\n"
    user_prompt += (
        f"Method checklist: {strategy['checklist']}\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += (
        "Solve independently with your assigned reasoning method. "
        "Produce a compact but complete solving process, then state the final answer. "
        "Do not wander through multiple dead-end approaches or repeat the same derivation in different forms. "
        "Stop once you have a decisive verification. "
        "For multiple-choice or symbolic tasks, final_answer must be only the shortest answer string. "
        "Return exactly one JSON object with keys \"final_answer\" and \"reasoning\". "
        "Keep the reasoning under 160 words and make it detailed enough for peer inspection in the next debate round. "
        "Return JSON only."
    )
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
    """Build the critique stage for the single-agent reflection baseline."""

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
        "Review the draft once. Identify the most concrete mistake, missing evidence, or unsupported jump if one exists. "
        "If the draft already looks sound, say that it is supported and name the strongest supporting check. "
        "Return plain text feedback only. Keep it under 120 words."
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
    """Build the revise stage for the single-agent reflection baseline."""

    _assert_prompt_version(prompt_version)
    guardrails = _multiple_choice_debate_guardrails(sample)
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
        "Revise only if the feedback identifies a concrete issue. "
        "If the feedback confirms the draft, keep the answer and tighten the justification. "
        "Return exactly one JSON object with keys \"final_answer\" and \"reasoning\". "
        "Keep the reasoning under 160 words and focus only on the decisive correction or confirmation. "
        "Return JSON only."
    )
    if guardrails:
        user_prompt += f"\nFor answer-format safety: {guardrails}"
    return [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": user_prompt},
    ]


def build_solution_selector_messages(
    sample: DatasetSample,
    candidate_solutions: list[dict[str, str | int]],
    *,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    """Build the shared best-solution selector for MAD-like methods."""

    _assert_prompt_version(prompt_version)
    candidate_block = "\n\n".join(
        "\n".join(
            [
                f"Candidate {item['agent_id']}:",
                f"- persona: {item['persona_name']}",
                f"- reasoning_method: {item['strategy_name']}",
                f"- solving_process: {str(item['reasoning']).strip()}",
                f"- final_answer: {str(item['answer']).strip()}",
            ]
        )
        for item in candidate_solutions
    )
    user_prompt = (
        "Choose the best final answer by comparing the candidate solutions and their solving processes.\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += (
        f"Candidate solutions:\n{candidate_block}\n\n"
        "Prefer the candidate with the strongest evidence, most coherent reasoning chain, and best match to the required answer format. "
        "Do not invent a new answer that no candidate supported. "
        "If every candidate is weak, pick the least flawed supported answer or abstain with selected_agent_id 4.\n"
        "Return exactly one JSON object with keys \"selected_agent_id\", \"final_answer\", and \"rationale\". "
        "selected_agent_id must be 1, 2, 3, or 4. Return JSON only."
    )
    return [
        {"role": "system", "content": _selector_system_prompt()},
        {"role": "user", "content": user_prompt},
    ]


def _system_prompt() -> str:
    return build_json_system_prompt(
        "You are one reasoning agent in a Diverse Multi-Agent Debate experiment.",
        extra_rules=[
            "Solve the task carefully using only the provided question and context.",
            "Respect your assigned persona and reasoning-method constraints.",
            "Make the reasoning field concise, decisive, and still substantive enough for peer review in later rounds.",
            "Prefer one clean derivation over multiple speculative branches.",
            "Do not add natural-language text before or after the JSON object.",
        ],
    )


def _plain_text_system_prompt() -> str:
    return "\n".join(
        [
            "You are one reasoning agent in a Diverse Multi-Agent Debate experiment.",
            "Return plain text only.",
            "Do not use markdown fences.",
            "Solve the task carefully using only the provided question, context, and debate history.",
            "Respect your assigned persona and reasoning-method constraints.",
            "Write one coherent solving process rather than several speculative branches.",
        ]
    )


def _answer_only_system_prompt() -> str:
    return build_json_system_prompt(
        "You are one reasoning agent in a Diverse Multi-Agent Debate experiment.",
        extra_rules=[
            'Return exactly one JSON object with key "final_answer".',
            "Do not add a reasoning field unless explicitly requested.",
            "Do not add natural-language text before or after the JSON object.",
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


def _selector_system_prompt() -> str:
    return build_json_system_prompt(
        "You are the final solution selector for a controlled Diverse Multi-Agent Debate experiment.",
        extra_rules=[
            "Choose among the provided candidates instead of inventing a new solution.",
            "Use selected_agent_id 4 only when none of the candidates is supportable enough to endorse directly.",
            "Do not add natural-language text before or after the JSON object.",
        ],
    )


def _dataset_instruction(sample: DatasetSample) -> str:
    if sample.dataset == "hotpotqa":
        return dataset_instruction_for_sample(sample, hotpot_style="shortest_span_copy")
    return dataset_instruction_for_sample(sample)


def _assert_prompt_version(prompt_version: str) -> None:
    if prompt_version != DEFAULT_PROMPT_VERSION:
        raise ValueError(f"Unsupported DMAD prompt_version: {prompt_version}")


def _strategy_spec(sample: DatasetSample, strategy_name: str) -> dict[str, str]:
    if strategy_name == "cot":
        return {
            "label": "CoT",
            "summary": "Chain-of-Thought prompting: solve the problem step by step.",
            "guidance": "Use one clean derivation, avoid unsupported recall, and end by checking that the chosen answer exactly matches the conclusion.",
            "checklist": (
                "write the intermediate reasoning in order, justify key transformations, "
                "and verify that the final answer matches the original question"
            ),
            "debate_action": (
                "compare the exact inference chain against the peers' chains and repair the first unsupported or inconsistent step"
            ),
        }
    if strategy_name == "sbp":
        return {
            "label": "SBP",
            "summary": "Step-Back Prompting: step back first, derive the governing concepts or principles, then solve the specific instance.",
            "guidance": "State the governing principle first, apply it to the options or equations, and prefer principle-based elimination over unsupported memory.",
            "checklist": (
                "identify the high-level principle first, map the question to that principle, "
                "and only then instantiate the concrete reasoning steps"
            ),
            "debate_action": (
                "use peer solutions to check whether your chosen principles are complete and whether a more suitable abstraction resolves the disagreement"
            ),
        }
    if strategy_name == "pot_l2m":
        if sample.dataset in {"math500", "gsm8k"}:
            return {
                "label": "PoT",
                "summary": "Program-of-Thoughts prompting: express the calculation as a concise Python-style program or symbolic procedure and use it to solve the problem.",
                "guidance": "Write a compact program-like calculation or symbolic procedure, compute the target quantity, and ensure the final answer is the computed result.",
                "checklist": (
                    "translate the calculation into explicit symbolic or Python-style steps, keep the program logic consistent, "
                    "and ensure the computed result corresponds exactly to the asked quantity"
                ),
                "debate_action": (
                    "inspect whether a peer reveals a missing variable, case split, or computational branch, then revise the program-like procedure accordingly"
                ),
            }
        return {
            "label": "L2M",
            "summary": "Least-to-Most prompting: decompose the problem into simpler subquestions and solve them in sequence.",
            "guidance": "Break the task into short subquestions, answer them in order, and use the subanswers to eliminate options without leaning on unsupported background claims.",
            "checklist": (
                "break the task into smaller subquestions, answer them in dependency order, "
                "and ensure the final answer follows from the accumulated sub-results"
            ),
            "debate_action": (
                "check whether peer solutions expose a missing subquestion or a wrong dependency between substeps, then rebuild the decomposition"
            ),
        }
    raise ValueError(f"Unsupported DMAD strategy: {strategy_name}")


def _multiple_choice_debate_guardrails(sample: DatasetSample) -> str:
    if sample.dataset not in {"mmlu_pro", "mmlu_abstract_algebra", "gpqa_diamond"}:
        return ""
    return (
        "do not switch options just because two peers agree; "
        "switch only if you can name one concrete flaw in your previous option and one concrete reason the new option is better supported; "
        "if a minority answer is the only one backed by an explicit calculation or option-by-option comparison, prefer the better-supported minority over unsupported consensus; "
        "if you compute a numeric result, map it to the matching option letter before returning; "
        'end the reasoning by explicitly stating "Best option: <LETTER>" and make final_answer match that letter exactly'
    )


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
            blocks.append(
                "\n".join(
                    [
                        "Your previous record:",
                        f"- method: {own.get('strategy_name', '')}",
                        f"- solving process: {str(own.get('reasoning') or '').strip()}",
                        f"- final answer: {str(own.get('answer') or '').strip()}",
                    ]
                )
            )
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
            blocks.append("\n".join(peer_lines))
    return "\n\n".join(blocks)
