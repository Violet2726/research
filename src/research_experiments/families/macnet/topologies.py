"""MacNet DAG 拓扑生成与统计。"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random


@dataclass(frozen=True)
class TopologySpec:
    """单次运行使用的 DAG 拓扑。"""

    topology_type: str
    direction_mode: str
    node_count: int
    edges: list[tuple[int, int]]
    source_nodes: list[int]
    sink_nodes: list[int]
    dag_depth: int
    edge_density: float
    average_out_degree: float
    average_in_degree: float


def build_topology(
    topology_type: str,
    *,
    node_count: int,
    direction_mode: str,
    seed: int,
) -> TopologySpec:
    """按拓扑类型构造可执行 DAG。"""

    if node_count < 1:
        raise ValueError("node_count must be positive.")
    if topology_type == "chain":
        edges = _build_chain_edges(node_count, direction_mode)
    elif topology_type == "star":
        edges = _build_star_edges(node_count, direction_mode)
    elif topology_type == "tree":
        edges = _build_tree_edges(node_count, direction_mode)
    elif topology_type == "mesh":
        edges = _build_mesh_edges(node_count, direction_mode)
    elif topology_type == "layer":
        edges = _build_layer_edges(node_count, direction_mode)
    elif topology_type == "random":
        edges = _build_random_edges(node_count, direction_mode, seed=seed)
    else:
        raise ValueError(f"Unsupported topology_type: {topology_type}")

    indegree = {node_id: 0 for node_id in range(node_count)}
    outdegree = {node_id: 0 for node_id in range(node_count)}
    for source, target in edges:
        outdegree[source] += 1
        indegree[target] += 1
    source_nodes = [node_id for node_id, value in indegree.items() if value == 0]
    sink_nodes = [node_id for node_id, value in outdegree.items() if value == 0]
    max_edges = max(1, node_count * (node_count - 1) / 2)
    return TopologySpec(
        topology_type=topology_type,
        direction_mode=direction_mode,
        node_count=node_count,
        edges=edges,
        source_nodes=source_nodes,
        sink_nodes=sink_nodes,
        dag_depth=_dag_depth(node_count, edges),
        edge_density=round(len(edges) / max_edges, 6),
        average_out_degree=round(sum(outdegree.values()) / node_count, 6),
        average_in_degree=round(sum(indegree.values()) / node_count, 6),
    )


def _build_chain_edges(node_count: int, direction_mode: str) -> list[tuple[int, int]]:
    if direction_mode == "divergent":
        return [(index, index + 1) for index in range(node_count - 1)]
    return [(index + 1, index) for index in range(node_count - 1)]


def _build_star_edges(node_count: int, direction_mode: str) -> list[tuple[int, int]]:
    if node_count == 1:
        return []
    if direction_mode == "divergent":
        return [(0, node_id) for node_id in range(1, node_count)]
    sink = node_count - 1
    return [(node_id, sink) for node_id in range(node_count - 1)]


def _build_tree_edges(node_count: int, direction_mode: str) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for parent in range(node_count):
        left = 2 * parent + 1
        right = 2 * parent + 2
        for child in (left, right):
            if child >= node_count:
                continue
            if direction_mode == "divergent":
                edges.append((parent, child))
            else:
                edges.append((child, parent))
    return edges


def _build_mesh_edges(node_count: int, direction_mode: str) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for source in range(node_count):
        for target in range(source + 1, node_count):
            edges.append((source, target) if direction_mode == "divergent" else (target, source))
    return edges


def _build_layer_edges(node_count: int, direction_mode: str) -> list[tuple[int, int]]:
    if node_count == 1:
        return []
    layer_count = max(2, int(math.log2(node_count)) + 1)
    if direction_mode == "divergent":
        layer_sizes = _allocate_layer_sizes(node_count, layer_count, single_head=True)
    else:
        layer_sizes = _allocate_layer_sizes(node_count, layer_count, single_head=False)
    layers: list[list[int]] = []
    cursor = 0
    for size in layer_sizes:
        layer = list(range(cursor, cursor + size))
        layers.append(layer)
        cursor += size
    if direction_mode != "divergent":
        layers = list(reversed(layers))
    edges: list[tuple[int, int]] = []
    for current_layer, next_layer in zip(layers, layers[1:]):
        for source in current_layer:
            for target in next_layer:
                edges.append((source, target))
    return edges


def _allocate_layer_sizes(node_count: int, layer_count: int, *, single_head: bool) -> list[int]:
    sizes = [1 if single_head else 0]
    remaining = node_count - sizes[0]
    tail_layers = layer_count - 1
    base = max(1, remaining // tail_layers) if tail_layers else remaining
    remainder = remaining
    for _ in range(tail_layers):
        size = min(base, remainder)
        sizes.append(size)
        remainder -= size
    index = 1
    while remainder > 0:
        sizes[index] += 1
        remainder -= 1
        index += 1
        if index >= len(sizes):
            index = 1
    return [size for size in sizes if size > 0]


def _build_random_edges(node_count: int, direction_mode: str, *, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    base_edges = {(index, index + 1) for index in range(node_count - 1)}
    candidates = [
        (source, target)
        for source in range(node_count)
        for target in range(source + 1, node_count)
        if (source, target) not in base_edges
    ]
    rng.shuffle(candidates)
    extra_edge_count = min(len(candidates), max(0, node_count - 2))
    for edge in candidates[:extra_edge_count]:
        base_edges.add(edge)
    edges = sorted(base_edges)
    if direction_mode == "divergent":
        return edges
    return [(target, source) for source, target in edges]


def _dag_depth(node_count: int, edges: list[tuple[int, int]]) -> int:
    incoming: dict[int, list[int]] = {node_id: [] for node_id in range(node_count)}
    outgoing: dict[int, list[int]] = {node_id: [] for node_id in range(node_count)}
    indegree = {node_id: 0 for node_id in range(node_count)}
    for source, target in edges:
        incoming[target].append(source)
        outgoing[source].append(target)
        indegree[target] += 1
    queue = sorted(node_id for node_id, value in indegree.items() if value == 0)
    order: list[int] = []
    while queue:
        node_id = queue.pop(0)
        order.append(node_id)
        for target in outgoing[node_id]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
                queue.sort()
    depth = [0 for _ in range(node_count)]
    for node_id in order:
        if incoming[node_id]:
            depth[node_id] = 1 + max(depth[parent] for parent in incoming[node_id])
    return max(depth, default=0)
