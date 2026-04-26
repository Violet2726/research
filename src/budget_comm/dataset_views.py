"""`same-context / split-context` 样本视图构造。

`budget_comm` 的关键前提之一是：不同 agent 不一定看到同一份上下文。
本模块负责把统一样本切分成 agent 级别的可见视图，并显式记录覆盖范围、
完整上下文哈希和分片标题，便于后续做泄漏校验、覆盖校验和机制分析。
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from budget_comm.config import ContextViewConfig
from experiment_core.datasets import DatasetSample


@dataclass(frozen=True)
class ContextView:
    """某个 agent 在单题上实际看到的上下文视图。"""

    dataset: str
    sample_id: str
    agent_id: int
    track_name: str
    question: str
    context_text: str
    view_kind: str
    includes_full_context: bool
    coverage_items: list[str]
    required_coverage_items: list[str]
    shard_titles: list[str]
    full_context_hash: str
    view_context_hash: str
    metadata: dict[str, Any]


def build_context_views(
    sample: DatasetSample,
    config: ContextViewConfig,
    *,
    agent_count: int,
) -> list[ContextView]:
    """按轨道配置为单题构造全部 agent 视图。

    当前 `budget_comm v1` 固定为 3 个 agent，因此这里也显式约束为三路视图设计。
    """
    if agent_count != 3:
        raise ValueError(f"budget_comm v1 only supports 3 agents, got {agent_count}.")
    if config.track_name == "same_context":
        return _build_same_context_views(sample, config)
    if sample.dataset == "strategyqa":
        return _build_strategyqa_split_views(sample, config)
    if sample.dataset == "hotpotqa":
        return _build_hotpotqa_split_views(sample, config)
    raise ValueError(f"Unsupported track={config.track_name!r} for dataset={sample.dataset!r}.")


def serialize_view_row(
    *,
    run_id: str,
    split_name: str,
    question_preview: str,
    view: ContextView,
) -> dict[str, Any]:
    """把视图对象转成可直接写入 JSONL 的稳定行结构。"""
    return {
        "run_id": run_id,
        "dataset": view.dataset,
        "split": split_name,
        "sample_id": view.sample_id,
        "question_preview": question_preview,
        "agent_id": view.agent_id,
        "track_name": view.track_name,
        "view_kind": view.view_kind,
        "includes_full_context": view.includes_full_context,
        "coverage_items": view.coverage_items,
        "required_coverage_items": view.required_coverage_items,
        "shard_titles": view.shard_titles,
        "full_context_hash": view.full_context_hash,
        "view_context_hash": view.view_context_hash,
        "context_text": view.context_text,
        "metadata": view.metadata,
    }


def _build_same_context_views(sample: DatasetSample, config: ContextViewConfig) -> list[ContextView]:
    """构造所有 agent 共享完整上下文的视图。"""
    full_context = sample.prompt_context.strip()
    full_hash = _stable_hash(full_context)
    return [
        ContextView(
            dataset=sample.dataset,
            sample_id=sample.sample_id,
            agent_id=agent_id,
            track_name=config.track_name,
            question=sample.question,
            context_text=full_context,
            view_kind="same_context",
            includes_full_context=True,
            coverage_items=[],
            required_coverage_items=[],
            shard_titles=[],
            full_context_hash=full_hash,
            view_context_hash=full_hash,
            metadata={},
        )
        for agent_id in range(1, 4)
    ]


def _build_strategyqa_split_views(sample: DatasetSample, config: ContextViewConfig) -> list[ContextView]:
    """按事实奇偶位与分解步骤为 StrategyQA 构造三路视图。"""
    if config.strategyqa_mode != "facts_even_odd_plus_decomposition":
        raise ValueError(f"Unsupported strategyqa_mode: {config.strategyqa_mode}")
    facts = [str(item).strip() for item in sample.metadata.get("facts", []) if str(item).strip()]
    decomposition = [str(item).strip() for item in sample.metadata.get("decomposition", []) if str(item).strip()]
    description = str(sample.metadata.get("description") or "").strip()
    even_position_facts = [fact for index, fact in enumerate(facts, start=1) if index % 2 == 0]
    odd_position_facts = [fact for index, fact in enumerate(facts, start=1) if index % 2 == 1]
    full_hash = _stable_hash(sample.prompt_context.strip())
    return [
        ContextView(
            dataset=sample.dataset,
            sample_id=sample.sample_id,
            agent_id=1,
            track_name=config.track_name,
            question=sample.question,
            context_text=_join_sections("Facts for agent_1", even_position_facts),
            view_kind="strategyqa_even_facts",
            includes_full_context=False,
            coverage_items=even_position_facts,
            required_coverage_items=facts,
            shard_titles=[],
            full_context_hash=full_hash,
            view_context_hash=_stable_hash(_join_sections("Facts for agent_1", even_position_facts)),
            metadata={"strategyqa_mode": config.strategyqa_mode},
        ),
        ContextView(
            dataset=sample.dataset,
            sample_id=sample.sample_id,
            agent_id=2,
            track_name=config.track_name,
            question=sample.question,
            context_text=_join_sections("Facts for agent_2", odd_position_facts),
            view_kind="strategyqa_odd_facts",
            includes_full_context=False,
            coverage_items=odd_position_facts,
            required_coverage_items=facts,
            shard_titles=[],
            full_context_hash=full_hash,
            view_context_hash=_stable_hash(_join_sections("Facts for agent_2", odd_position_facts)),
            metadata={"strategyqa_mode": config.strategyqa_mode},
        ),
        ContextView(
            dataset=sample.dataset,
            sample_id=sample.sample_id,
            agent_id=3,
            track_name=config.track_name,
            question=sample.question,
            context_text=_render_strategyqa_decomposition(description, decomposition),
            view_kind="strategyqa_decomposition_description",
            includes_full_context=False,
            coverage_items=[],
            required_coverage_items=facts,
            shard_titles=[],
            full_context_hash=full_hash,
            view_context_hash=_stable_hash(_render_strategyqa_decomposition(description, decomposition)),
            metadata={"strategyqa_mode": config.strategyqa_mode},
        ),
    ]


def _build_hotpotqa_split_views(sample: DatasetSample, config: ContextViewConfig) -> list[ContextView]:
    """按 supporting paragraphs 与干扰段落为 HotpotQA 构造三路视图。"""
    if config.hotpotqa_mode != "supporting_paragraph_shards":
        raise ValueError(f"Unsupported hotpotqa_mode: {config.hotpotqa_mode}")
    raw_context = sample.metadata.get("raw_context") or {}
    supporting_facts = sample.metadata.get("supporting_facts") or {}
    paragraph_map = _hotpot_paragraph_map(raw_context)
    all_titles = [title for title, _ in paragraph_map]
    supporting_titles = []
    for title in supporting_facts.get("title", []):
        normalized = str(title).strip()
        if normalized and normalized not in supporting_titles:
            supporting_titles.append(normalized)
    distractor_titles = [title for title in all_titles if title not in supporting_titles]
    agent_1_titles = [item for item in [*_slice(supporting_titles, 0, 1), *_slice(distractor_titles, 0, 1)] if item]
    agent_2_titles = [item for item in [*_slice(supporting_titles, 1, 2), *_slice(distractor_titles, 1, 2)] if item]
    agent_3_titles = distractor_titles[2:]
    title_list_block = "Available titles:\n" + "\n".join(f"- {title}" for title in all_titles) if all_titles else ""
    full_hash = _stable_hash(sample.prompt_context.strip())
    agent_1_context = _render_hotpot_shard(paragraph_map, agent_1_titles, title_list=None)
    agent_2_context = _render_hotpot_shard(paragraph_map, agent_2_titles, title_list=None)
    agent_3_context = _render_hotpot_shard(paragraph_map, agent_3_titles, title_list=title_list_block)
    return [
        ContextView(
            dataset=sample.dataset,
            sample_id=sample.sample_id,
            agent_id=1,
            track_name=config.track_name,
            question=sample.question,
            context_text=agent_1_context,
            view_kind="hotpot_supporting_plus_distractor_a",
            includes_full_context=False,
            coverage_items=[title for title in agent_1_titles if title in supporting_titles],
            required_coverage_items=supporting_titles,
            shard_titles=agent_1_titles,
            full_context_hash=full_hash,
            view_context_hash=_stable_hash(agent_1_context),
            metadata={"hotpotqa_mode": config.hotpotqa_mode},
        ),
        ContextView(
            dataset=sample.dataset,
            sample_id=sample.sample_id,
            agent_id=2,
            track_name=config.track_name,
            question=sample.question,
            context_text=agent_2_context,
            view_kind="hotpot_supporting_plus_distractor_b",
            includes_full_context=False,
            coverage_items=[title for title in agent_2_titles if title in supporting_titles],
            required_coverage_items=supporting_titles,
            shard_titles=agent_2_titles,
            full_context_hash=full_hash,
            view_context_hash=_stable_hash(agent_2_context),
            metadata={"hotpotqa_mode": config.hotpotqa_mode},
        ),
        ContextView(
            dataset=sample.dataset,
            sample_id=sample.sample_id,
            agent_id=3,
            track_name=config.track_name,
            question=sample.question,
            context_text=agent_3_context,
            view_kind="hotpot_remaining_distractors_plus_titles",
            includes_full_context=False,
            coverage_items=[],
            required_coverage_items=supporting_titles,
            shard_titles=agent_3_titles,
            full_context_hash=full_hash,
            view_context_hash=_stable_hash(agent_3_context),
            metadata={"hotpotqa_mode": config.hotpotqa_mode},
        ),
    ]


def _render_strategyqa_decomposition(description: str, decomposition: list[str]) -> str:
    """把 StrategyQA 的描述与分解步骤渲染成第三路视图文本。"""
    sections: list[str] = []
    if description:
        sections.append(f"Description:\n{description}")
    if decomposition:
        sections.append(_join_sections("Decomposition steps", decomposition))
    return "\n\n".join(section for section in sections if section).strip()


def _hotpot_paragraph_map(raw_context: dict[str, Any]) -> list[tuple[str, str]]:
    """把 HotpotQA 原始上下文整理为 `(标题, 段落文本)` 列表。"""
    titles = raw_context.get("title", [])
    sentences = raw_context.get("sentences", [])
    rendered: list[tuple[str, str]] = []
    for title, paragraph_sentences in zip(titles, sentences, strict=False):
        normalized_title = str(title).strip()
        joined = " ".join(str(sentence).strip() for sentence in paragraph_sentences if str(sentence).strip())
        rendered.append((normalized_title, f"[{normalized_title}] {joined}".strip()))
    return rendered


def _render_hotpot_shard(
    paragraph_map: list[tuple[str, str]],
    selected_titles: list[str],
    *,
    title_list: str | None,
) -> str:
    """渲染单个 HotpotQA 视图分片，并按需附加标题清单。"""
    rendered_sections = [paragraph for title, paragraph in paragraph_map if title in selected_titles]
    if title_list:
        rendered_sections.append(title_list)
    return "\n\n".join(section for section in rendered_sections if section).strip()


def _join_sections(header: str, items: list[str]) -> str:
    """把一个条目列表渲染为带标题的文本块。"""
    if not items:
        return ""
    return f"{header}:\n" + "\n".join(f"- {item}" for item in items)


def _slice(items: list[str], start: int, stop: int) -> list[str]:
    """返回去空后的稳定切片。"""
    return [item for item in items[start:stop] if item]


def _stable_hash(text: str) -> str:
    """为上下文文本计算稳定哈希。"""
    return sha256(text.encode("utf-8")).hexdigest()


def asdict_view(view: ContextView) -> dict[str, Any]:
    """提供给测试与调试使用的 dataclass 序列化。"""
    return asdict(view)
