"""Restricted (k, delta)-LDP augmentation and residual graph construction."""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Any, Callable, Optional

import networkx as nx

from .graph_utils import EdgeRef, add_edge_with_attrs, edge_by_id, ensure_edge_ids, outgoing_edges, remove_edge_by_id


@dataclass
class LDPResult:
    feasible: bool
    paths: list[list[int]]
    residual: nx.MultiDiGraph
    base_weight: float
    used_cost: int
    remaining_budget: int
    message: str


def bounded_cost_shortest_path(
    R: nx.MultiDiGraph,
    s: Any,
    t: Any,
    budget: int,
    path_validator: Optional[Callable[[list[int]], bool]] = None,
) -> Optional[tuple[list[int], float, int]]:
    """Find a minimum-weight s-t path with integer cost at most budget.

    This is a Bellman-Ford style dynamic program over (node, cost). It supports
    negative residual weights as long as the best simple bounded-cost path is
    well-defined; paths are capped at |V|-1 edge relaxations for augmentation.
    """
    nodes = list(R.nodes)
    negative_cost_floor = sum(min(0, int(data.get("cost", 0))) for _, _, _, data in R.edges(keys=True, data=True))
    min_cost = negative_cost_floor
    max_cost = int(budget)
    if min_cost > max_cost:
        return None
    width = max_cost - min_cost + 1
    offset = -min_cost
    inf = float("inf")
    dist: dict[Any, list[float]] = {v: [inf] * width for v in nodes}
    pred: dict[tuple[Any, int], tuple[Any, int, int]] = {}
    dist[s][offset] = 0.0
    for _ in range(max(len(nodes) - 1, 0)):
        changed = False
        old = {v: vals[:] for v, vals in dist.items()}
        for u in nodes:
            for idx0, w0 in enumerate(old[u]):
                if w0 == inf:
                    continue
                c0 = idx0 - offset
                for edge in outgoing_edges(R, u):
                    c1 = c0 + edge.cost
                    idx1 = c1 + offset
                    if min_cost <= c1 <= max_cost and w0 + edge.weight < dist[edge.v][idx1]:
                        dist[edge.v][idx1] = w0 + edge.weight
                        pred[(edge.v, c1)] = (u, c0, edge.eid)
                        changed = True
        if not changed:
            break
    feasible_costs = range(min_cost, max_cost + 1)
    sink_costs = sorted(feasible_costs, key=lambda c: dist[t][c + offset])
    if dist[t][sink_costs[0] + offset] == inf:
        return None
    for best_cost in sink_costs:
        if dist[t][best_cost + offset] == inf:
            break
        edge_ids: list[int] = []
        cur = (t, best_cost)
        while cur[0] != s:
            prev_u, prev_c, eid = pred[cur]
            edge_ids.append(eid)
            cur = (prev_u, prev_c)
        edge_ids.reverse()
        if path_validator is None or path_validator(edge_ids):
            return edge_ids, dist[t][best_cost + offset], best_cost

    if path_validator is None:
        return None
    return _bounded_cost_shortest_valid_path_by_dfs(R, s, t, budget, path_validator)


def _bounded_cost_shortest_valid_path_by_dfs(
    R: nx.MultiDiGraph,
    s: Any,
    t: Any,
    budget: int,
    path_validator: Callable[[list[int]], bool],
) -> Optional[tuple[list[int], float, int]]:
    """Fallback enumerator used when the DP-best path violates final-flow constraints.

    The DP table stores only one predecessor per state, so it cannot recover the
    second-best path for the same state. This DFS fallback preserves correctness
    for small residual graphs and hand-built counterexamples by searching simple
    residual paths and validating the resulting original-flow decomposition.
    """
    nodes = list(R.nodes)
    min_cost = sum(min(0, int(data.get("cost", 0))) for _, _, _, data in R.edges(keys=True, data=True))
    best: Optional[tuple[list[int], float, int]] = None
    max_depth = max(len(nodes) - 1, 0)

    def dfs(cur: Any, seen: set[Any], path: list[int], weight: float, cost: int) -> None:
        nonlocal best
        if len(path) > max_depth:
            return
        if cost > budget or cost < min_cost:
            return
        if best is not None and weight >= best[1]:
            return
        if cur == t:
            if path_validator(path):
                best = (path[:], weight, cost)
            return
        edges = sorted(outgoing_edges(R, cur), key=lambda e: (e.weight, e.cost, e.eid))
        for edge in edges:
            if edge.v in seen:
                continue
            seen.add(edge.v)
            path.append(edge.eid)
            dfs(edge.v, seen, path, weight + edge.weight, cost + edge.cost)
            path.pop()
            seen.remove(edge.v)

    dfs(s, {s}, [], 0.0, 0)
    return best


def _base_node(node: Any) -> Any:
    if isinstance(node, tuple) and len(node) == 2 and node[1] in {"in", "out"}:
        return node[0]
    return node


def _is_split_label(node: Any) -> bool:
    return isinstance(node, tuple) and len(node) == 2 and node[1] in {"in", "out"}


def _readd_edge(R: nx.MultiDiGraph, edge: EdgeRef, u: Any, v: Any) -> None:
    add_edge_with_attrs(
        R,
        u,
        v,
        eid=edge.eid,
        weight=edge.weight,
        cost=edge.cost,
        is_reverse=edge.is_reverse,
        is_split_edge=edge.is_split_edge,
        original_eid=edge.original_eid,
        desc=edge.desc,
    )


def _mark_split_nodes(R: nx.MultiDiGraph, path_edges: list[int], s: Any, t: Any) -> None:
    """Physically split first-used internal nodes into v_in/v_out.

    Incoming residual edges are rewired to v_in, outgoing residual edges are
    rewired from v_out, and the internal edge v_in -> v_out has cost 1. The
    current augmenting path is found before splitting, so the first visit to a
    node is free; future paths must cross the split edge and spend budget.
    """
    by_id = edge_by_id(R)
    touched: set[Any] = set()
    for eid in path_edges:
        edge = by_id[eid]
        for node in (edge.u, edge.v):
            base = _base_node(node)
            if base not in (s, t) and not _is_split_label(node):
                touched.add(base)
    for v in touched:
        if R.nodes[v].get("split_done"):
            continue
        vin = (v, "in")
        vout = (v, "out")
        R.add_node(vin)
        R.add_node(vout)
        current_edges = list(edge_by_id(R).values())
        for edge in current_edges:
            if edge.is_split_edge:
                continue
            new_u = vout if edge.u == v else edge.u
            new_v = vin if edge.v == v else edge.v
            if new_u != edge.u or new_v != edge.v:
                remove_edge_by_id(R, edge.eid)
                _readd_edge(R, edge, new_u, new_v)
        add_edge_with_attrs(R, vin, vout, weight=0.0, cost=1, is_split_edge=True, desc=f"split {v} in->out")
        add_edge_with_attrs(R, vout, vin, weight=0.0, cost=0, is_split_edge=True, desc=f"split {v} out->in")
        R.nodes[v]["split_done"] = True


def update_restricted_residual(R: nx.MultiDiGraph, path_edges: list[int], s: Any, t: Any) -> None:
    """Reverse each chosen path edge and add split bookkeeping for internal nodes."""
    _mark_split_nodes(R, path_edges, s, t)
    for eid in path_edges:
        edge = remove_edge_by_id(R, eid)
        add_edge_with_attrs(
            R,
            edge.v,
            edge.u,
            weight=-edge.weight,
            cost=-edge.cost,
            is_reverse=not edge.is_reverse,
            is_split_edge=edge.is_split_edge,
            original_eid=edge.original_eid if edge.original_eid is not None else edge.eid,
            desc=f"reverse of {edge.eid}",
        )


def _path_flow_delta(R: nx.MultiDiGraph, path_edges: list[int]) -> dict[int, int]:
    by_id = edge_by_id(R)
    delta: dict[int, int] = defaultdict(int)
    for eid in path_edges:
        edge = by_id[eid]
        if edge.is_split_edge:
            continue
        original_eid = edge.original_eid if edge.original_eid is not None else edge.eid
        delta[int(original_eid)] += -1 if edge.is_reverse else 1
    return dict(delta)


def _apply_flow_delta(flow: dict[int, int], delta: dict[int, int]) -> dict[int, int]:
    out = dict(flow)
    for eid, change in delta.items():
        out[eid] = out.get(eid, 0) + change
        if out[eid] == 0:
            del out[eid]
    return out


def _decompose_original_flow(
    base_edges: dict[int, tuple[Any, Any, float]],
    flow: dict[int, int],
    s: Any,
    t: Any,
    k: int,
) -> Optional[list[list[int]]]:
    adjacency: dict[Any, list[tuple[Any, int]]] = defaultdict(list)
    for eid, amount in flow.items():
        if amount < 0 or amount > 1:
            return None
        if amount == 0:
            continue
        u, v, _weight = base_edges[eid]
        adjacency[u].append((v, eid))

    paths: list[list[int]] = []
    for _ in range(k):
        found = _find_flow_path(adjacency, s, t)
        if found is None:
            return None
        path_edges: list[int] = []
        for u, index in found:
            _v, eid = adjacency[u].pop(index)
            path_edges.append(eid)
        paths.append(path_edges)
    if any(adjacency.values()):
        return None
    return paths


def _find_flow_path(
    adjacency: dict[Any, list[tuple[Any, int]]],
    s: Any,
    t: Any,
) -> Optional[list[tuple[Any, int]]]:
    stack: list[tuple[Any, list[tuple[Any, int]], set[Any]]] = [(s, [], {s})]
    while stack:
        node, path, seen = stack.pop()
        if node == t:
            return path
        for index, (next_node, _eid) in enumerate(adjacency.get(node, [])):
            if next_node in seen:
                continue
            stack.append((next_node, path + [(node, index)], seen | {next_node}))
    return None


def _common_node_count(
    base_edges: dict[int, tuple[Any, Any, float]],
    paths: list[list[int]],
    s: Any,
    t: Any,
) -> Optional[int]:
    counts: dict[Any, int] = defaultdict(int)
    for path in paths:
        nodes = [s]
        cur = s
        for eid in path:
            u, v, _weight = base_edges[eid]
            if u != cur:
                return None
            nodes.append(v)
            cur = v
        if cur != t:
            return None
        for node in set(nodes[1:-1]):
            counts[node] += 1
    if any(count > 2 for count in counts.values()):
        return None
    return sum(1 for count in counts.values() if count >= 2)


def _project_residual_path_nodes(R: nx.MultiDiGraph, path_edges: list[int]) -> Optional[list[Any]]:
    """Project a residual edge path to original node labels, collapsing split nodes."""
    if not path_edges:
        return []
    by_id = edge_by_id(R)
    first = by_id[path_edges[0]]
    projected = [_base_node(first.u)]
    cur = first.u
    for eid in path_edges:
        edge = by_id[eid]
        if edge.u != cur:
            return None
        cur = edge.v
        node = _base_node(edge.v)
        if projected[-1] != node:
            projected.append(node)
    return projected


def _common_count_from_node_paths(paths: list[list[Any]], s: Any, t: Any) -> Optional[int]:
    counts: dict[Any, int] = defaultdict(int)
    for path in paths:
        if not path or path[0] != s or path[-1] != t:
            return None
        internal = path[1:-1]
        if len(internal) != len(set(internal)):
            return None
        for node in set(internal):
            counts[node] += 1
    if any(count > 2 for count in counts.values()):
        return None
    return sum(1 for count in counts.values() if count >= 2)


def _flow_weight(base_edges: dict[int, tuple[Any, Any, float]], flow: dict[int, int]) -> float:
    return sum(base_edges[eid][2] * amount for eid, amount in flow.items() if amount > 0)


def run_ldp(G: nx.MultiDiGraph, s: Any, t: Any, k: int, delta: int) -> LDPResult:
    """Run the restricted augmentation algorithm for k link-disjoint s-t paths."""
    R = ensure_edge_ids(G)
    for _, _, _, data in R.edges(keys=True, data=True):
        data["cost"] = int(data.get("cost", 0))
        if "weight" not in data:
            data["weight"] = 0.0
    base_edges = {
        int(data["eid"]): (u, v, float(data.get("weight", 0.0)))
        for u, v, _key, data in R.edges(keys=True, data=True)
    }
    q = int(delta)
    paths: list[list[int]] = []
    node_paths: list[list[Any]] = []
    base_weight = 0.0
    used_cost = 0
    for i in range(k):
        def validator(path_edges: list[int], round_index: int = i) -> bool:
            projected = _project_residual_path_nodes(R, path_edges)
            if projected is None:
                return False
            common_count = _common_count_from_node_paths(node_paths + [projected], s, t)
            return common_count is not None and common_count <= delta

        found = bounded_cost_shortest_path(R, s, t, q, path_validator=validator)
        if found is None:
            return LDPResult(False, paths, R, base_weight, used_cost, q, "no bounded-cost augmenting path")
        edge_ids, weight, cost = found
        projected = _project_residual_path_nodes(R, edge_ids)
        if projected is None:
            return LDPResult(False, paths, R, base_weight, used_cost, q, "residual path is not directed")
        new_common_count = _common_count_from_node_paths(node_paths + [projected], s, t)
        if new_common_count is None or new_common_count > delta:
            return LDPResult(False, paths, R, base_weight, used_cost, q, "path set violates common-node budget")
        paths.append(edge_ids)
        node_paths.append(projected)
        used_cost = new_common_count
        q = delta - used_cost
        base_weight += weight
        update_restricted_residual(R, edge_ids, s, t)
    return LDPResult(True, paths, R, base_weight, used_cost, q, "ok")
