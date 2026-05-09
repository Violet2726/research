"""HotpotQA split-context 视图构造。

该模块为通信必要性实验提供受控的上下文拆分：
两个 agent 分别看到不同 supporting paragraph，另一个 agent 主要看到干扰项与标题清单，
从而构造“没有通信就难以完整整合证据”的实验下界。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any

from experiment_core.foundation.datasets import DatasetSample


@dataclass(frozen=True)
class HotpotView:
    """单个 agent 在 HotpotQA 样本上可见的上下文。"""

    dataset: str
    sample_id: str
    agent_id: int
    view_kind: str
    question: str
    context_text: str
    includes_full_context: bool
    shard_titles: list[str]
    coverage_titles: list[str]
    required_titles: list[str]
    full_context_hash: str
    view_context_hash: str
    title_inventory: list[str]
    metadata: dict[str, Any]


def build_hotpot_views(sample: DatasetSample) -> list[HotpotView]:
    """构造 3 个 split 视图和 1 个 full-context 参考视图。"""
    paragraph_map = _paragraph_map(sample.metadata.get("raw_context") or {})
    all_titles = [title for title, _ in paragraph_map]
    required_titles = _supporting_titles(sample.metadata.get("supporting_facts") or {})
    distractor_titles = [title for title in all_titles if title not in required_titles]
    full_context = _render_paragraphs(paragraph_map, all_titles, include_title_inventory=False)
    full_hash = _stable_hash(full_context)

    # agent 1/2 各自拿一个 gold supporting paragraph，再配少量 distractor；
    # agent 3 只拿 distractor 和标题清单，用来形成“无通信难整合”的下界。
    agent_1_titles = [*_slice(required_titles, 0, 1), *_slice(distractor_titles, 0, 3)]
    agent_2_titles = [*_slice(required_titles, 1, 2), *_slice(distractor_titles, 3, 6)]
    used = set(agent_1_titles + agent_2_titles)
    agent_3_titles = [title for title in all_titles if title not in used]
    split_specs = [
        (1, "supporting_shard_a", agent_1_titles, [title for title in agent_1_titles if title in required_titles], False),
        (2, "supporting_shard_b", agent_2_titles, [title for title in agent_2_titles if title in required_titles], False),
        (3, "distractor_titles_only", agent_3_titles, [], True),
    ]
    views: list[HotpotView] = []
    for agent_id, view_kind, shard_titles, coverage_titles, include_inventory in split_specs:
        context_text = _render_paragraphs(paragraph_map, shard_titles, include_title_inventory=include_inventory, title_inventory=all_titles)
        views.append(
            HotpotView(
                dataset=sample.dataset,
                sample_id=sample.sample_id,
                agent_id=agent_id,
                view_kind=view_kind,
                question=sample.question,
                context_text=context_text,
                includes_full_context=False,
                shard_titles=shard_titles,
                coverage_titles=coverage_titles,
                required_titles=required_titles,
                full_context_hash=full_hash,
                view_context_hash=_stable_hash(context_text),
                title_inventory=all_titles if include_inventory else [],
                metadata={"type": sample.metadata.get("type"), "level": sample.metadata.get("level")},
            )
        )
    views.append(
        HotpotView(
            dataset=sample.dataset,
            sample_id=sample.sample_id,
            agent_id=0,
            view_kind="full_context",
            question=sample.question,
            context_text=full_context,
            includes_full_context=True,
            shard_titles=all_titles,
            coverage_titles=required_titles,
            required_titles=required_titles,
            full_context_hash=full_hash,
            view_context_hash=full_hash,
            title_inventory=[],
            metadata={"type": sample.metadata.get("type"), "level": sample.metadata.get("level")},
        )
    )
    return views


def serialize_view_row(*, run_id: str, split_name: str, view: HotpotView) -> dict[str, Any]:
    """写入 sample_views.jsonl 的稳定行格式。"""
    return {
        "run_id": run_id,
        "dataset": view.dataset,
        "split": split_name,
        "sample_id": view.sample_id,
        "agent_id": view.agent_id,
        "view_kind": view.view_kind,
        "includes_full_context": view.includes_full_context,
        "shard_titles": view.shard_titles,
        "coverage_titles": view.coverage_titles,
        "required_titles": view.required_titles,
        "full_context_hash": view.full_context_hash,
        "view_context_hash": view.view_context_hash,
        "title_inventory": view.title_inventory,
        "context_text": view.context_text,
        "metadata": view.metadata,
    }


def asdict_view(view: HotpotView) -> dict[str, Any]:
    """测试使用的 dataclass 序列化。"""
    return asdict(view)


def _paragraph_map(raw_context: dict[str, Any]) -> list[tuple[str, list[str]]]:
    titles = raw_context.get("title", [])
    paragraphs = raw_context.get("sentences", [])
    rendered: list[tuple[str, list[str]]] = []
    for title, sentences in zip(titles, paragraphs, strict=False):
        normalized_title = str(title).strip()
        sentence_list = [str(sentence).strip() for sentence in sentences if str(sentence).strip()]
        if normalized_title:
            rendered.append((normalized_title, sentence_list))
    return rendered


def _supporting_titles(raw_supporting_facts: dict[str, Any]) -> list[str]:
    titles: list[str] = []
    for title in raw_supporting_facts.get("title", []):
        normalized = str(title).strip()
        if normalized and normalized not in titles:
            titles.append(normalized)
    return titles


def _render_paragraphs(
    paragraph_map: list[tuple[str, list[str]]],
    selected_titles: list[str],
    *,
    include_title_inventory: bool,
    title_inventory: list[str] | None = None,
) -> str:
    selected = set(selected_titles)
    sections: list[str] = []
    for title, sentences in paragraph_map:
        if title not in selected:
            continue
        lines = [f"[{title}]"]
        for sent_id, sentence in enumerate(sentences):
            lines.append(f"({sent_id}) {sentence}")
        sections.append("\n".join(lines))
    if include_title_inventory and title_inventory:
        sections.append("Available paragraph titles:\n" + "\n".join(f"- {title}" for title in title_inventory))
    return "\n\n".join(section for section in sections if section).strip()


def _slice(items: list[str], start: int, stop: int) -> list[str]:
    return [item for item in items[start:stop] if item]


def _stable_hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


