"""DoG 原论文高保真复现使用的提示词与文本解析。"""

from __future__ import annotations

import re
from typing import Any


FILTER_BEST_REL_PROMPT = """
For a multi-constraint problem, solving it requires first answering the initial question and then constraining the answer to the initial question to get the correct answer.
Given a multi-constraint problem and multiple relations, choose the best relations that are most helpful in answering the initial sub-question.
The number and content of the selected relations must be given, for example: relation_2: some.relation.
Choose only the most relevant relation or a very small set of ties.
""".strip()


ENOUGH_ANSWER_PROMPT = """
Given a question and the associated retrieved knowledge graph triples, decide whether the information is sufficient to answer the question with these triples and your knowledge.
Be sure to type Yes or No first.
If the triples are enough, explain briefly and give the answer using the expressions that appear in the triples.
If they are not enough, explain briefly why the current triples are insufficient.
""".strip()


SIMPLIFIER_TEMPLATE = """
The triple [e, r, e'] describes a fact used to answer a sub-question of a multi-hop problem, resulting in a smaller question that contains the answer entity and removes the resolved hop.

The formal process is:
Given an N-hop problem Q_N:
- the problem contains a core entity and a target relation
- the known triples resolve one sub-question
- the resolved answer entity becomes the core entity of the new Q_(N-1) problem

Here is your task:
[N-hop question]
{question}
[known information]
{reasoning_triples}
Here's your discussion history:
{chat_history}
{role_description}
{final_prompt}
""".strip()


SIMPLIFIER_ROLE_DESCRIPTIONS = {
    "Question Simplifying Expert": (
        "You are an expert in problem simplification. Use the known triples to answer one sub-question "
        "of the original N-hop problem and produce a smaller remaining question."
    ),
    "Critic": (
        "You are a serious critic. Compare the original N-hop question and the simplified question in the chat history. "
        "If they are effectively the same, point out the failure and give the corrected simplified question. "
        "Make sure the simplified question now centers on the known answer entity."
    ),
    "Linguist": (
        "You are a linguist who removes redundant wording and incomplete simplification. "
        "Your output must only contain the final simplified question in the format `simplified question: ...`."
    ),
}


SIMPLIFIER_FINAL_PROMPTS = {
    "Question Simplifying Expert": "",
    "Critic": "Double check error cases where simplification was not successful.",
    "Linguist": (
        "You just need to output the correct simplified problem. "
        "Do not print the original question. The output format must be `simplified question: ...`."
    ),
}


def build_relation_selection_messages(question: str, relation_mapping: dict[str, str]) -> list[dict[str, str]]:
    relation_block = ", ".join(f"{key}: {value}" for key, value in relation_mapping.items())
    return [
        {"role": "system", "content": "You are a careful knowledge-graph reasoning assistant."},
        {
            "role": "user",
            "content": f"{FILTER_BEST_REL_PROMPT}\nQ:{question}\nRelations:{relation_block}\nA:",
        },
    ]


def parse_selected_relations(response_text: str, relation_mapping: dict[str, str], *, limit: int) -> list[str]:
    """按官方脚本风格解析 `relation_x` 或显式 relation 名。"""

    selected: list[str] = []
    for match in re.findall(r"relation_\d+", response_text, flags=re.IGNORECASE):
        relation = relation_mapping.get(match.lower())
        if relation and relation not in selected:
            selected.append(relation)
    if not selected:
        lowered = response_text.lower()
        for relation in relation_mapping.values():
            if relation.lower() in lowered and relation not in selected:
                selected.append(relation)
    return selected[:limit]


def build_enough_answer_messages(question: str, reasoning_triples: list[str]) -> list[dict[str, str]]:
    triple_block = ", ".join(f"path_{index}: {triple}" for index, triple in enumerate(reasoning_triples, start=1))
    return [
        {"role": "system", "content": "You are a careful knowledge-graph reasoning assistant."},
        {
            "role": "user",
            "content": f"{ENOUGH_ANSWER_PROMPT}\nQ:{question}\nKnowledge Triples:{triple_block}",
        },
    ]


def parse_enough_answer_output(response_text: str) -> dict[str, str]:
    normalized = str(response_text or "").strip()
    decision_match = re.search(r"\b(Yes|No)\b", normalized, flags=re.IGNORECASE)
    decision = decision_match.group(1).lower() if decision_match else "unknown"
    braced = [item.strip() for item in re.findall(r"\{([^{}]+)\}", normalized) if item.strip()]
    answer_text = ""
    for item in reversed(braced):
        if item.lower() not in {"yes", "no"}:
            answer_text = item
            break
    if not answer_text:
        answer_match = re.search(
            r"(?:answer\s*(?:is|:)\s*|therefore[, ]+the answer\s*(?:is|:)\s*)(?P<answer>.+)",
            normalized,
            flags=re.IGNORECASE,
        )
        if answer_match:
            answer_text = answer_match.group("answer").strip().rstrip(".")
    return {
        "decision": decision,
        "answer_text": answer_text,
        "raw_text": normalized,
    }


def build_simplifier_messages(
    *,
    question: str,
    reasoning_triples: list[str],
    role_name: str,
    chat_history: list[dict[str, str]],
) -> list[dict[str, str]]:
    history_block = "\n".join(f"{item['role']}: {item['content']}" for item in chat_history) if chat_history else "No prior discussion."
    triple_block = "\n".join(f"- {item}" for item in reasoning_triples) if reasoning_triples else "- No triples."
    role_description = SIMPLIFIER_ROLE_DESCRIPTIONS[role_name]
    final_prompt = SIMPLIFIER_FINAL_PROMPTS[role_name]
    prompt = SIMPLIFIER_TEMPLATE.format(
        question=question,
        reasoning_triples=triple_block,
        chat_history=history_block,
        role_description=role_description,
        final_prompt=final_prompt,
    )
    return [
        {"role": "system", "content": "You are a helpful assistant for multi-hop KGQA simplification."},
        {"role": "user", "content": prompt},
    ]


def parse_simplified_question(response_text: str, original_question: str) -> str:
    """从三角色链路的最终输出中提取 `Q_(N-1)`。"""

    text = str(response_text or "").strip()
    if not text:
        return ""
    matches = re.findall(r"simplified question\s*:\s*(.+)", text, flags=re.IGNORECASE)
    candidate = matches[-1].strip() if matches else text.splitlines()[-1].strip()
    candidate = candidate.strip().strip('"').strip("'")
    if not candidate:
        return ""
    if _normalize_question(candidate) == _normalize_question(original_question):
        return ""
    return candidate


def build_direct_answer_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You are a careful question answering assistant."},
        {"role": "user", "content": f"Q:{question}"},
    ]


def validate_plain_text_output(assistant_text: str, provider_reasoning_text: str) -> dict[str, Any]:
    """把普通文本输出包装成统一结构。"""

    text = str(assistant_text or "").strip() or str(provider_reasoning_text or "").strip()
    if not text:
        raise ValueError("Missing plain-text output.")
    return {"text": text}


def _normalize_question(question: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", question.lower()).split())
