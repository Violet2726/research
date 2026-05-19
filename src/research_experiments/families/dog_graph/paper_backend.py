"""DoG 原论文高保真复现使用的图后端。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import json
import os
import re
from typing import Any

import httpx

from research_experiments.core.config import BenchmarkConfig
from research_experiments.core.data.datasets import resolve_dataset_source_path


_FREEBASE_RELATION_BLACKLIST = {
    "type.object.type",
    "type.object.name",
    "user.narphorium.people.nndb_person.nndb_id",
}


@dataclass(frozen=True)
class EntityRef:
    """图后端中的实体引用。"""

    entity_id: str
    label: str


@dataclass(frozen=True)
class BackendCheckResult:
    """单个后端检查结果。"""

    backend_name: str
    dataset_slug: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class LocalEdge:
    """局部 KG 中的一条边。"""

    source_id: str
    source_label: str
    relation: str
    target_id: str
    target_label: str


@dataclass
class LocalReducedFreebaseSession:
    """按单题构造的局部 Freebase 会话。"""

    sample_id: str
    topic_entities: list[EntityRef]
    adjacency: dict[str, dict[str, list[EntityRef]]]
    reverse_adjacency: dict[str, dict[str, list[EntityRef]]]
    triple_index: dict[tuple[str, str], list[str]]
    backend_name: str = "local_reduced_freebase"
    debug: dict[str, Any] = field(default_factory=dict)

    def list_relations(self, head_entities: list[EntityRef]) -> list[str]:
        relations: set[str] = set()
        for entity in head_entities:
            relations.update(self.adjacency.get(entity.entity_id, {}).keys())
            relations.update(self.adjacency.get(entity.label, {}).keys())
        return sorted(relations)

    def expand_relation(self, head_entities: list[EntityRef], relation: str) -> list[EntityRef]:
        expanded: list[EntityRef] = []
        for entity in head_entities:
            expanded.extend(self.adjacency.get(entity.entity_id, {}).get(relation, []))
            expanded.extend(self.adjacency.get(entity.label, {}).get(relation, []))
        return _unique_entities(expanded)

    def build_reasoning_triples(self, head_entities: list[EntityRef], relation: str) -> list[str]:
        triples: list[str] = []
        for entity in head_entities:
            triples.extend(self.triple_index.get((entity.entity_id, relation), []))
            triples.extend(self.triple_index.get((entity.label, relation), []))
        return list(dict.fromkeys(triples))[:12]

    def validate(self, dataset_slug: str) -> BackendCheckResult:
        return BackendCheckResult(self.backend_name, dataset_slug, True, "按题构造的局部 Freebase 会话可用。")

    def for_sample(self, sample) -> LocalReducedFreebaseSession:
        return self


class FreebaseSparqlBackend:
    """基于本地 Virtuoso 的 Freebase 检索后端。"""

    backend_name = "freebase_virtuoso"

    def __init__(self, sparql_url: str) -> None:
        self.sparql_url = sparql_url
        self._client = httpx.Client(timeout=30.0)
        self._name_cache: dict[str, str] = {}

    def close(self) -> None:
        self._client.close()

    def validate(self, dataset_slug: str) -> BackendCheckResult:
        try:
            rows = self._run_query(
                "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?relation WHERE { ns:m.03_r3 ?relation ?x . } LIMIT 1"
            )
        except Exception as exc:
            return BackendCheckResult(self.backend_name, dataset_slug, False, str(exc))
        if not rows:
            return BackendCheckResult(self.backend_name, dataset_slug, False, "Virtuoso 可访问，但查询未返回任何关系。")
        return BackendCheckResult(self.backend_name, dataset_slug, True, f"查询成功，返回 {len(rows)} 条结果。")

    def for_sample(self, sample) -> FreebaseSparqlBackend:
        return self

    def list_relations(self, head_entities: list[EntityRef]) -> list[str]:
        relations: set[str] = set()
        for entity in head_entities:
            query = (
                "PREFIX ns: <http://rdf.freebase.com/ns/>\n"
                "SELECT ?relation\n"
                "WHERE {\n"
                f"  ns:{entity.entity_id} ?relation ?x .\n"
                "}"
            )
            rows = self._run_query(query)
            for row in rows:
                relation = str(row.get("relation", {}).get("value") or "").replace("http://rdf.freebase.com/ns/", "")
                if not relation or self._abandon_relation(relation):
                    continue
                relations.add(relation)
        return sorted(relations)

    def expand_relation(self, head_entities: list[EntityRef], relation: str) -> list[EntityRef]:
        expanded: list[EntityRef] = []
        for entity in head_entities:
            query = (
                "PREFIX ns: <http://rdf.freebase.com/ns/>\n"
                "SELECT ?tailEntity\n"
                "WHERE {\n"
                f"  ns:{entity.entity_id} ns:{relation} ?tailEntity .\n"
                "}"
            )
            rows = self._run_query(query)
            for row in rows:
                target_id = str(row.get("tailEntity", {}).get("value") or "").replace("http://rdf.freebase.com/ns/", "")
                if not target_id:
                    continue
                expanded.append(EntityRef(target_id, self.entity_name(target_id)))
        return _unique_entities(expanded)

    def build_reasoning_triples(self, head_entities: list[EntityRef], relation: str) -> list[str]:
        triples: list[str] = []
        for entity in head_entities:
            for tail in self.expand_relation([entity], relation):
                if tail.label and tail.label != "UnName_Entity":
                    triples.append(f"({entity.label}, {relation}, {tail.label})")
                    continue
                for second_relation in self.list_relations([tail])[:3]:
                    second_hop = self.expand_relation([tail], second_relation)
                    named_targets = [item for item in second_hop if item.label and item.label != "UnName_Entity"]
                    if named_targets:
                        triples.append(f"({entity.label}, {relation}, {tail.entity_id}, {second_relation}, {named_targets[0].label})")
                        break
        return triples

    def entity_name(self, entity_id: str) -> str:
        cached = self._name_cache.get(entity_id)
        if cached is not None:
            return cached
        query = (
            "PREFIX ns: <http://rdf.freebase.com/ns/>\n"
            "SELECT DISTINCT ?tailEntity\n"
            "WHERE {\n"
            "  {\n"
            f"    ?entity ns:type.object.name ?tailEntity .\n"
            f"    FILTER(?entity = ns:{entity_id})\n"
            "  }\n"
            "  UNION\n"
            "  {\n"
            f"    ?entity <http://www.w3.org/2002/07/owl#sameAs> ?tailEntity .\n"
            f"    FILTER(?entity = ns:{entity_id})\n"
            "  }\n"
            "}\n"
            "LIMIT 1"
        )
        rows = self._run_query(query)
        if not rows:
            self._name_cache[entity_id] = "UnName_Entity"
            return "UnName_Entity"
        label = str(rows[0].get("tailEntity", {}).get("value") or "").strip() or "UnName_Entity"
        self._name_cache[entity_id] = label
        return label

    def _run_query(self, query: str) -> list[dict[str, Any]]:
        params = {
            "query": query,
            "format": "application/sparql-results+json",
        }
        response = self._client.get(self.sparql_url, params=params, headers={"Accept": "application/sparql-results+json"})
        response.raise_for_status()
        payload = response.json()
        return list(payload.get("results", {}).get("bindings", []))

    def _abandon_relation(self, relation: str) -> bool:
        if relation in _FREEBASE_RELATION_BLACKLIST:
            return True
        return relation.startswith(("common.", "kg.", "freebase.")) or "sameAs" in relation


class MetaqaGraphBackend:
    """基于 `kb.txt` 的轻量 MetaQA 图后端。"""

    backend_name = "metaqa_kb"

    def __init__(self, kb_path: str | Path) -> None:
        self.kb_path = Path(kb_path)
        self.graph: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        self._loaded = False

    def validate(self, dataset_slug: str) -> BackendCheckResult:
        if not self.kb_path.exists():
            return BackendCheckResult(self.backend_name, dataset_slug, False, f"缺少 MetaQA 图文件：{self.kb_path}")
        self._ensure_loaded()
        if not self.graph:
            return BackendCheckResult(self.backend_name, dataset_slug, False, "MetaQA 图文件存在，但未解析出任何三元组。")
        return BackendCheckResult(self.backend_name, dataset_slug, True, f"已加载 {len(self.graph)} 个头实体。")

    def for_sample(self, sample) -> MetaqaGraphBackend:
        return self

    def list_relations(self, head_entities: list[EntityRef]) -> list[str]:
        self._ensure_loaded()
        relations: set[str] = set()
        for entity in head_entities:
            relations.update(self.graph.get(entity.label, {}).keys())
        relations.discard("~release_year")
        return sorted(relations)

    def expand_relation(self, head_entities: list[EntityRef], relation: str) -> list[EntityRef]:
        self._ensure_loaded()
        tail_entities: list[EntityRef] = []
        for entity in head_entities:
            for label in self.graph.get(entity.label, {}).get(relation, []):
                tail_entities.append(EntityRef(label, label))
        return _unique_entities(tail_entities)

    def build_reasoning_triples(self, head_entities: list[EntityRef], relation: str) -> list[str]:
        triples: list[str] = []
        for entity in head_entities:
            targets = self.graph.get(entity.label, {}).get(relation, [])
            for target in targets:
                if relation.startswith("~"):
                    triples.append(f"({target}, {relation[1:]}, {entity.label})")
                else:
                    triples.append(f"({entity.label}, {relation}, {target})")
        return list(dict.fromkeys(triples))[:12]

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if not self.kb_path.exists():
            self._loaded = True
            return
        with self.kb_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = line.strip()
                if not row:
                    continue
                head, relation, tail = row.split("|", 2)
                self.graph[head][relation].append(tail)
                self.graph[tail][f"~{relation}"].append(head)
        self._loaded = True


class LocalReducedFreebaseBackend:
    """资源受限机器上的局部 Freebase 后端。"""

    backend_name = "local_reduced_freebase"

    def __init__(self) -> None:
        self._webquestions_index: dict[str, dict[str, Any]] | None = None

    def validate(self, dataset_slug: str) -> BackendCheckResult:
        if dataset_slug == "webquestions_paper_test":
            dataset_root = resolve_dataset_source_path("webquestions")
            required = [
                dataset_root / "test.json",
                dataset_root / "relation_paths_test.json",
                dataset_root / "branched_relation_paths_test.json",
                dataset_root / "freebase_key_test.json",
            ]
            missing = [path.as_posix() for path in required if not path.exists()]
            if missing:
                return BackendCheckResult(self.backend_name, dataset_slug, False, f"缺少 WebQuestions 局部后端注释：{', '.join(missing)}")
        return BackendCheckResult(self.backend_name, dataset_slug, True, "局部 KG 后端资产齐全。")

    def for_sample(self, sample) -> LocalReducedFreebaseSession:
        topic_entities = _sample_topic_entities(sample)
        if sample.dataset == "webquestions_paper_test":
            session = self._build_webquestions_session(sample, topic_entities)
        elif sample.dataset in {"webqsp", "cwq"}:
            session = self._build_sparql_session(sample, topic_entities)
        elif sample.dataset == "grailqa_test":
            session = self._build_grailqa_session(sample, topic_entities)
        else:
            session = LocalReducedFreebaseSession(
                sample_id=sample.sample_id,
                topic_entities=topic_entities,
                adjacency=defaultdict(dict),
                reverse_adjacency=defaultdict(dict),
                triple_index={},
                debug={"dataset": sample.dataset},
            )
        return session

    def _build_webquestions_session(self, sample, topic_entities: list[EntityRef]) -> LocalReducedFreebaseSession:
        record = self._match_webquestions_annotation(sample)
        relation_paths = list(record.get("branched_relation_paths") or []) or list(record.get("relation_paths") or [])
        answer_label = _first_answer_label(sample)
        edges: list[LocalEdge] = []
        for path_index, path_spec in enumerate(relation_paths[:8], start=1):
            relation_chain = [str(item).strip() for item in (path_spec[0] if path_spec else []) if str(item).strip()]
            if not relation_chain:
                continue
            previous_id = topic_entities[0].entity_id if topic_entities else "topic:0"
            previous_label = topic_entities[0].label if topic_entities else "topic"
            for hop_index, relation in enumerate(relation_chain, start=1):
                is_last = hop_index == len(relation_chain)
                target_id = f"{sample.sample_id}:path{path_index}:hop{hop_index}"
                target_label = answer_label if is_last else f"path_{path_index}_hop_{hop_index}"
                edges.append(LocalEdge(previous_id, previous_label, relation, target_id, target_label))
                previous_id = target_id
                previous_label = target_label
        if not edges:
            relation = "candidate_answer_relation"
            target_label = answer_label or "?answer"
            edges.append(LocalEdge(topic_entities[0].entity_id, topic_entities[0].label, relation, f"{sample.sample_id}:answer", target_label))
        decoy_relations = []
        for item in sample.metadata.get("freebase_mids", []):
            if isinstance(item, dict) and str(item.get("concept") or "").strip():
                decoy_relations.append(f"linked_entity::{str(item.get('concept')).strip()}")
        return _build_local_session(sample.sample_id, topic_entities, edges, extra_debug={"decoy_relations": decoy_relations, "annotation": record})

    def _build_sparql_session(self, sample, topic_entities: list[EntityRef]) -> LocalReducedFreebaseSession:
        source_record = sample.metadata.get("source_record", {}) or {}
        query = str(source_record.get("sparql") or source_record.get("Sparql") or "").strip()
        if not query and sample.dataset == "webqsp":
            parses = list(source_record.get("Parses") or [])
            if parses and isinstance(parses[0], dict):
                query = str(parses[0].get("Sparql") or "").strip()
        answer_alias = _first_answer_label(sample)
        selected_variables = _extract_selected_variables(query)
        parsed_triples = _extract_sparql_triples(query)
        edges: list[LocalEdge] = []
        node_labels: dict[str, str] = {entity.entity_id: entity.label for entity in topic_entities}
        record_topic_map = dict(sample.metadata.get("topic_entity") or {})
        for entity_id, entity_label in record_topic_map.items():
            node_labels[f"ns:{entity_id}"] = str(entity_label).strip()
            node_labels[entity_id] = str(entity_label).strip()
        for subject, relation, obj in parsed_triples:
            subject_id = _normalize_sparql_node_id(subject)
            object_id = _normalize_sparql_node_id(obj)
            subject_label = _label_for_sparql_node(subject, node_labels, answer_alias, selected_variables)
            object_label = _label_for_sparql_node(obj, node_labels, answer_alias, selected_variables)
            node_labels[subject_id] = subject_label
            node_labels[object_id] = object_label
            edges.append(LocalEdge(subject_id, subject_label, relation, object_id, object_label))
        return _build_local_session(sample.sample_id, topic_entities, edges, extra_debug={"sparql": query, "selected_variables": sorted(selected_variables)})

    def _build_grailqa_session(self, sample, topic_entities: list[EntityRef]) -> LocalReducedFreebaseSession:
        graph_query = sample.metadata.get("graph_query") or {}
        nodes = list(graph_query.get("nodes") or [])
        edges_payload = list(graph_query.get("edges") or [])
        answer_alias = _first_answer_label(sample)
        node_labels: dict[str, str] = {}
        node_ids: dict[int, str] = {}
        answer_anchor_ids: set[str] = set()
        for node in nodes:
            raw_id = str(node.get("id") or f"node:{node.get('nid')}").strip()
            node_id = raw_id or f"node:{node.get('nid')}"
            label = str(node.get("friendly_name") or node.get("id") or node_id).strip()
            if bool(node.get("question_node")) and answer_alias:
                label = answer_alias
                answer_anchor_ids.add(node_id)
            node_labels[node_id] = label
            try:
                node_ids[int(node.get("nid"))] = node_id
            except Exception:
                pass
        built_edges: list[LocalEdge] = []
        for edge in edges_payload:
            source_id = node_ids.get(int(edge.get("start")), f"node:{edge.get('start')}")
            target_id = node_ids.get(int(edge.get("end")), f"node:{edge.get('end')}")
            built_edges.append(
                LocalEdge(
                    source_id,
                    node_labels.get(source_id, source_id),
                    str(edge.get("relation") or edge.get("friendly_name") or "").strip(),
                    target_id,
                    node_labels.get(target_id, target_id),
                )
            )
        return _build_local_session(sample.sample_id, topic_entities, built_edges, extra_debug={"answer_anchors": sorted(answer_anchor_ids)})

    def _match_webquestions_annotation(self, sample) -> dict[str, Any]:
        if self._webquestions_index is None:
            self._webquestions_index = _build_webquestions_annotation_index()
        key = _normalize_question_key(sample.question)
        return dict(self._webquestions_index.get(key, {}))


def build_backend_for_benchmark(
    benchmark: BenchmarkConfig,
    *,
    freebase_sparql_url: str,
    freebase_backend_mode: str,
) -> FreebaseSparqlBackend | MetaqaGraphBackend | LocalReducedFreebaseBackend:
    """按 benchmark slug 构造对应的论文级图后端。"""

    if benchmark.slug.startswith("metaqa_"):
        kb_path = resolve_dataset_source_path("metaqa/kb.txt")
        return MetaqaGraphBackend(kb_path)
    if str(freebase_backend_mode).strip().lower() == "virtuoso":
        return FreebaseSparqlBackend(os.getenv("RESEARCH_FREEBASE_SPARQL_URL", freebase_sparql_url))
    return LocalReducedFreebaseBackend()


def validate_required_backends(
    benchmarks: list[BenchmarkConfig],
    *,
    freebase_sparql_url: str,
    freebase_backend_mode: str,
) -> dict[str, Any]:
    """检查实验所需后端与官方资产是否齐全。"""

    seen: set[tuple[str, str]] = set()
    checks: list[dict[str, Any]] = []
    backends: list[FreebaseSparqlBackend | MetaqaGraphBackend | LocalReducedFreebaseBackend] = []
    try:
        for benchmark in benchmarks:
            source_path = resolve_dataset_source_path(benchmark.source_path)
            source_ok = source_path.exists()
            checks.append(
                {
                    "backend_name": "dataset_asset",
                    "dataset_slug": benchmark.slug,
                    "ok": source_ok,
                    "detail": source_path.as_posix() if source_ok else f"缺少数据文件：{source_path.as_posix()}",
                }
            )
            backend = build_backend_for_benchmark(
                benchmark,
                freebase_sparql_url=freebase_sparql_url,
                freebase_backend_mode=freebase_backend_mode,
            )
            key = (benchmark.slug, backend.__class__.__name__)
            if key in seen:
                continue
            seen.add(key)
            backends.append(backend)
            result = backend.validate(benchmark.slug)
            checks.append(
                {
                    "backend_name": result.backend_name,
                    "dataset_slug": result.dataset_slug,
                    "ok": result.ok,
                    "detail": result.detail,
                }
            )
    finally:
        for backend in backends:
            close = getattr(backend, "close", None)
            if callable(close):
                close()
    return {
        "ok": all(bool(item["ok"]) for item in checks),
        "checks": checks,
    }


def _build_local_session(
    sample_id: str,
    topic_entities: list[EntityRef],
    edges: list[LocalEdge],
    *,
    extra_debug: dict[str, Any] | None = None,
) -> LocalReducedFreebaseSession:
    adjacency: dict[str, dict[str, list[EntityRef]]] = defaultdict(lambda: defaultdict(list))
    reverse: dict[str, dict[str, list[EntityRef]]] = defaultdict(lambda: defaultdict(list))
    triple_index: dict[tuple[str, str], list[str]] = defaultdict(list)
    for edge in edges:
        target = EntityRef(edge.target_id, edge.target_label)
        adjacency[edge.source_id][edge.relation].append(target)
        adjacency[edge.source_label][edge.relation].append(target)
        triple_index[(edge.source_id, edge.relation)].append(f"({edge.source_label}, {edge.relation}, {edge.target_label})")
        triple_index[(edge.source_label, edge.relation)].append(f"({edge.source_label}, {edge.relation}, {edge.target_label})")
        source = EntityRef(edge.source_id, edge.source_label)
        reverse_relation = f"~{edge.relation}"
        adjacency[edge.target_id][reverse_relation].append(source)
        adjacency[edge.target_label][reverse_relation].append(source)
        triple_index[(edge.target_id, reverse_relation)].append(f"({edge.target_label}, {reverse_relation[1:]}, {edge.source_label})")
        triple_index[(edge.target_label, reverse_relation)].append(f"({edge.target_label}, {reverse_relation[1:]}, {edge.source_label})")
    return LocalReducedFreebaseSession(
        sample_id=sample_id,
        topic_entities=topic_entities,
        adjacency=adjacency,
        reverse_adjacency=reverse,
        triple_index=triple_index,
        debug=extra_debug or {},
    )


def _sample_topic_entities(sample) -> list[EntityRef]:
    payload = dict(sample.metadata.get("topic_entity") or {})
    if payload:
        return [EntityRef(str(entity_id).strip(), str(entity_label).strip()) for entity_id, entity_label in payload.items()]
    topic_id = str(sample.metadata.get("topic_entity_id") or sample.metadata.get("topic_entity_name") or sample.sample_id).strip()
    topic_label = str(sample.metadata.get("topic_entity_name") or topic_id).strip()
    return [EntityRef(topic_id, topic_label)]


def _first_answer_label(sample) -> str:
    answers = sample.metadata.get("answers")
    if isinstance(answers, list):
        for answer in answers:
            normalized = str(answer).strip()
            if normalized:
                return normalized
    try:
        decoded = json.loads(sample.reference_answer)
    except Exception:
        return str(sample.reference_answer).strip()
    if isinstance(decoded, list):
        for answer in decoded:
            normalized = str(answer).strip()
            if normalized:
                return normalized
    return str(sample.reference_answer).strip()


def _extract_selected_variables(query: str) -> set[str]:
    matches = re.findall(r"\?([A-Za-z0-9_]+)\s+AS\s+\?value", query, flags=re.IGNORECASE)
    if matches:
        return {f"?{item}" for item in matches}
    direct = re.findall(r"SELECT(?:\s+DISTINCT)?\s+(\?[A-Za-z0-9_]+)", query, flags=re.IGNORECASE)
    return set(direct)


def _extract_sparql_triples(query: str) -> list[tuple[str, str, str]]:
    pattern = re.compile(
        r"(?P<subj>\?[A-Za-z0-9_]+|ns:[^\s]+)\s+(?P<pred>ns:[^\s]+)\s+(?P<obj>\?[A-Za-z0-9_]+|ns:[^\s]+|\"[^\"]+\"(?:@[A-Za-z\-]+)?)\s*\.",
        flags=re.MULTILINE,
    )
    triples: list[tuple[str, str, str]] = []
    for match in pattern.finditer(query):
        predicate = str(match.group("pred")).replace("ns:", "").strip()
        if not predicate or predicate in _FREEBASE_RELATION_BLACKLIST:
            continue
        triples.append((match.group("subj").strip(), predicate, match.group("obj").strip()))
    return triples


def _normalize_sparql_node_id(token: str) -> str:
    token = str(token).strip()
    if token.startswith("ns:"):
        return token[3:]
    return token


def _label_for_sparql_node(token: str, node_labels: dict[str, str], answer_alias: str, selected_variables: set[str]) -> str:
    normalized = _normalize_sparql_node_id(token)
    if token in selected_variables and answer_alias:
        return answer_alias
    if normalized in node_labels:
        return node_labels[normalized]
    if token.startswith("ns:"):
        return normalized.replace(".", " ").replace("_", " ")
    if token.startswith("?"):
        return token
    if token.startswith("\""):
        return token.strip("\"")
    return normalized


def _build_webquestions_annotation_index() -> dict[str, dict[str, Any]]:
    dataset_root = resolve_dataset_source_path("webquestions")
    payload = json.loads((dataset_root / "test.json").read_text(encoding="utf-8"))
    relation_paths = _load_annotation_rows(dataset_root / "relation_paths_test.json")
    branched_paths = _load_annotation_rows(dataset_root / "branched_relation_paths_test.json")
    freebase_mids = _load_annotation_rows(dataset_root / "freebase_mids_test.json")
    rows: dict[str, dict[str, Any]] = {}
    for record in payload:
        question = str(record.get("qText") or "").strip()
        sample_id = str(record.get("qId") or "").strip()
        if not question or not sample_id:
            continue
        rows[_normalize_question_key(question)] = {
            "sample_id": sample_id,
            "relation_paths": relation_paths.get(sample_id, {}).get("relPaths", []),
            "branched_relation_paths": branched_paths.get(sample_id, {}).get("relPaths", []),
            "freebase_mids": freebase_mids.get(sample_id, {}).get("freebaseMids", []),
        }
    return rows


def _load_annotation_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: dict[str, dict[str, Any]] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        sample_id = str(item.get("qId") or "").strip()
        if sample_id:
            rows[sample_id] = item
    return rows


def _normalize_question_key(question: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", question.lower()).split())


def _unique_entities(entities: list[EntityRef]) -> list[EntityRef]:
    seen: set[tuple[str, str]] = set()
    unique: list[EntityRef] = []
    for entity in entities:
        key = (entity.entity_id, entity.label)
        if key in seen:
            continue
        seen.add(key)
        unique.append(entity)
    return unique
