from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReasoningMethodSpec:
    label: str
    summary: str
    guidance: str
    checklist: str
    debate_action: str


def resolve_reasoning_method(dataset: str, method_name: str) -> ReasoningMethodSpec:
    normalized = _normalize_method_name(dataset, method_name)
    if normalized == "cot":
        return ReasoningMethodSpec(
            label="CoT",
            summary="Chain-of-Thought prompting: derive the answer through an explicit step-by-step solution.",
            guidance="Write one coherent derivation, keep every decisive inference explicit, and verify the final conclusion before committing.",
            checklist="state the decisive steps in order, justify the key transformation, and check that the conclusion answers the original question",
            debate_action="compare your earliest unsupported step against peer solutions and revise only when a concrete flaw is exposed",
        )
    if normalized == "sbp":
        return ReasoningMethodSpec(
            label="SBP",
            summary="Step-Back Prompting: step back to the governing concept or principle before instantiating the concrete solution.",
            guidance="Name the governing concept first, then map the concrete problem onto that concept and solve only with the necessary details.",
            checklist="identify the governing concept, explain why it applies here, and instantiate the minimal concrete steps from that abstraction",
            debate_action="use peer solutions to test whether your abstraction is incomplete or whether a better governing principle explains the task",
        )
    if normalized == "pot":
        return ReasoningMethodSpec(
            label="PoT",
            summary="Program-of-Thoughts prompting: solve the task by writing executable Python and storing the result in ans.",
            guidance="Translate the decisive computation into executable Python, store the final result in ans, and make the answer follow the executed result.",
            checklist="define variables explicitly, cover the necessary branches or case splits, store the final computed value in ans, and verify that ans answers the original question",
            debate_action="inspect whether peers expose a missing branch, variable, constraint, or case split, then repair the executable program accordingly",
        )
    if normalized == "l2m":
        return ReasoningMethodSpec(
            label="L2M",
            summary="Least-to-Most prompting: decompose the task into smaller subquestions and solve them in dependency order.",
            guidance="Break the problem into short subquestions, solve them from easiest to hardest, and derive the final answer only from completed subresults.",
            checklist="list the smallest necessary subquestions, solve them in dependency order, and carry forward only the evidence needed for the final answer",
            debate_action="check whether peer solutions reveal a missing subquestion or a wrong dependency between your substeps",
        )
    raise ValueError(f"Unsupported reasoning method: {method_name}")


def normalize_reasoning_method_name(dataset: str, method_name: str) -> str:
    return _normalize_method_name(dataset, method_name)


def _normalize_method_name(dataset: str, method_name: str) -> str:
    normalized = str(method_name or "").strip().lower()
    if normalized in {"cot", "chain_of_thought"}:
        return "cot"
    if normalized in {"sbp", "step_back"}:
        return "sbp"
    if normalized in {"pot", "program_of_thoughts"}:
        return "pot" if dataset in {"math500", "competition_math", "gsm8k"} else "l2m"
    if normalized in {"l2m", "least_to_most"}:
        return "l2m"
    if normalized in {"pot_l2m"}:
        return "pot" if dataset in {"math500", "competition_math", "gsm8k"} else "l2m"
    if normalized.endswith("_sc"):
        return _normalize_method_name(dataset, normalized[: -len("_sc")])
    return normalized
