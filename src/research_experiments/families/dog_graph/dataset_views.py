"""DoG 复现使用的静态候选子图视角切片。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from research_experiments.core.data.datasets import DatasetSample


@dataclass(frozen=True)
class GraphView:
    """单个 agent 可见的图视角。"""

    agent_id: int
    view_kind: str
    context_text: str
    node_ids: list[str]
    node_labels: list[str]
    edge_keys: list[str]
    node_count: int
    edge_count: int
    visible_triples: list[str]
    structured_triples: list[tuple[str, str, str]]


def build_full_graph_view(sample: DatasetSample) -> GraphView:
    """构造单智能体基线使用的全图视角。"""

    graph = _graph_payload(sample)
    visible_triples = [_triple_text(edge) for edge in graph.get("edges", [])]
    return GraphView(
        agent_id=1,
        view_kind="full_subgraph",
        context_text=sample.prompt_context,
        node_ids=[str(node.get("id")) for node in graph.get("nodes", [])],
        node_labels=[str(node.get("label") or "") for node in graph.get("nodes", []) if str(node.get("label") or "").strip()],
        edge_keys=[_edge_key(edge) for edge in graph.get("edges", [])],
        node_count=len(graph.get("nodes", [])),
        edge_count=len(graph.get("edges", [])),
        visible_triples=visible_triples,
        structured_triples=[_structured_triple(edge) for edge in graph.get("edges", [])],
    )


def build_full_graph_views(sample: DatasetSample, agent_count: int) -> list[GraphView]:
    """为无通信多采样基线复制完整图视角。"""

    base = build_full_graph_view(sample)
    return [
        GraphView(
            agent_id=agent_id,
            view_kind=base.view_kind,
            context_text=base.context_text,
            node_ids=list(base.node_ids),
            node_labels=list(base.node_labels),
            edge_keys=list(base.edge_keys),
            node_count=base.node_count,
            edge_count=base.edge_count,
            visible_triples=list(base.visible_triples),
            structured_triples=list(base.structured_triples),
        )
        for agent_id in range(1, agent_count + 1)
    ]


def build_graph_views(sample: DatasetSample) -> list[GraphView]:
    """把统一候选子图切成三路固定视角。"""

    graph = _graph_payload(sample)
    edges = list(graph.get("edges", []))
    nodes = list(graph.get("nodes", []))
    relation_edges = [edge for edge in edges if str(edge.get("relation") or "") not in {"linked_entity", "question_mention"}]
    relation_edges = relation_edges or edges[:]
    entity_nodes = [
        node
        for node in nodes
        if str(node.get("type") or "") in {"linked_entity", "question_mention", "entity", "topic_seed"}
    ] or nodes[:]
    entity_node_ids = {str(node.get("id")) for node in entity_nodes}
    entity_edges = [
        edge
        for edge in edges
        if str(edge.get("source")) in entity_node_ids or str(edge.get("target")) in entity_node_ids
    ] or edges[:]
    summary_edges = sorted(edges, key=lambda edge: (-int(edge.get("support") or 0), str(edge.get("friendly_name") or edge.get("relation") or "")))[:6]
    global_snapshot = _render_global_snapshot(graph)

    return [
        _build_view(
            graph=graph,
            agent_id=1,
            view_kind="relation_path_view",
            selected_edges=relation_edges[:6],
            header_lines=[
                "Graph view: relation paths",
                "You can inspect the full graph snapshot below, but your primary responsibility is to follow relation chains that connect the topic seed to the answer slot.",
                "Prefer canonical KB-style titles over paraphrases or shortened forms.",
                "",
                global_snapshot,
            ],
        ),
        _build_view(
            graph=graph,
            agent_id=2,
            view_kind="entity_neighborhood_view",
            selected_edges=entity_edges[:6],
            selected_nodes=entity_nodes[:8],
            header_lines=[
                "Graph view: entity neighborhood",
                "You can inspect the full graph snapshot below, but your primary responsibility is to use linked entities, question clues, and local neighborhoods to disambiguate named entities.",
                "Prefer the current/canonical person, organization, language, or currency title rather than a loose paraphrase.",
                "",
                global_snapshot,
            ],
        ),
        _build_view(
            graph=graph,
            agent_id=3,
            view_kind="evidence_summary_view",
            selected_edges=summary_edges,
            header_lines=[
                "Graph view: evidence summary",
                "You can inspect the full graph snapshot below, but your primary responsibility is to synthesize the highest-support triples, domains, and query sketch into the most canonical answer title.",
                _graph_summary_line(graph),
                "",
                global_snapshot,
            ],
        ),
    ]


def _build_view(
    *,
    graph: dict[str, Any],
    agent_id: int,
    view_kind: str,
    selected_edges: list[dict[str, Any]],
    header_lines: list[str],
    selected_nodes: list[dict[str, Any]] | None = None,
) -> GraphView:
    node_lookup = {str(node.get("id")): node for node in graph.get("nodes", [])}
    node_ids = {str(node.get("id")) for node in selected_nodes or []}
    for edge in selected_edges:
        node_ids.add(str(edge.get("source")))
        node_ids.add(str(edge.get("target")))
    ordered_nodes = [node_lookup[node_id] for node_id in node_lookup]
    visible_triples = [_triple_text(edge) for edge in selected_edges]
    lines = [line for line in header_lines if line]
    lines.extend(["", "Visible nodes:"])
    lines.extend(f"- {node.get('id')}: {node.get('label')} [{node.get('type')}]" for node in ordered_nodes)
    lines.extend(["", "Visible triples / path fragments:"])
    lines.extend(f"- {triple}" for triple in visible_triples)
    return GraphView(
        agent_id=agent_id,
        view_kind=view_kind,
        context_text="\n".join(lines).strip(),
        node_ids=[str(node.get("id")) for node in ordered_nodes],
        node_labels=[str(node.get("label") or "") for node in ordered_nodes if str(node.get("label") or "").strip()],
        edge_keys=[_edge_key(edge) for edge in graph.get("edges", [])],
        node_count=len(ordered_nodes),
        edge_count=len(graph.get("edges", [])),
        visible_triples=visible_triples,
        structured_triples=[_structured_triple(edge) for edge in selected_edges],
    )


def _graph_payload(sample: DatasetSample) -> dict[str, Any]:
    payload = sample.metadata.get("candidate_subgraph")
    if isinstance(payload, dict):
        return payload
    return {"nodes": [], "edges": []}


def _graph_summary_line(graph: dict[str, Any]) -> str:
    domains = [str(item).strip() for item in graph.get("domains", []) if str(item).strip()]
    level = str(graph.get("level") or "").strip()
    parts = [
        f"topic_seed={graph.get('topic_seed')}" if graph.get("topic_seed") else "",
        f"domains={','.join(domains)}" if domains else "",
        f"level={level}" if level else "",
    ]
    return "Graph summary: " + "; ".join(part for part in parts if part)


def _render_global_snapshot(graph: dict[str, Any]) -> str:
    lines = ["Global graph snapshot:"]
    for edge in graph.get("edges", []):
        lines.append(f"- {_triple_text(edge)}")
    s_expression = str(graph.get("s_expression") or "").strip()
    if s_expression:
        lines.append(f"- query_sketch: {s_expression}")
    return "\n".join(lines)


def _triple_text(edge: dict[str, Any]) -> str:
    source = edge.get("source_label") or edge.get("source")
    target = edge.get("target_label") or edge.get("target")
    relation = edge.get("friendly_name") or edge.get("relation")
    return f"({source}, {relation}, {target})"


def _edge_key(edge: dict[str, Any]) -> str:
    return f"{edge.get('source')}|{edge.get('relation')}|{edge.get('target')}"


def _structured_triple(edge: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(edge.get("source_label") or edge.get("source") or "").strip(),
        str(edge.get("friendly_name") or edge.get("relation") or "").strip(),
        str(edge.get("target_label") or edge.get("target") or "").strip(),
    )
