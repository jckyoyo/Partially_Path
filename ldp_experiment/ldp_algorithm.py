"""Restricted (k, delta)-LDP augmentation and residual graph construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

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
    best_cost = min(feasible_costs, key=lambda c: dist[t][c + offset])
    if dist[t][best_cost + offset] == inf:
        return None
    edge_ids: list[int] = []
    cur = (t, best_cost)
    while cur[0] != s:
        prev_u, prev_c, eid = pred[cur]
        edge_ids.append(eid)
        cur = (prev_u, prev_c)
    edge_ids.reverse()
    return edge_ids, dist[t][best_cost + offset], best_cost


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


def run_ldp(G: nx.MultiDiGraph, s: Any, t: Any, k: int, delta: int) -> LDPResult:
    """Run the restricted augmentation algorithm for k link-disjoint s-t paths."""
    R = ensure_edge_ids(G)
    for _, _, _, data in R.edges(keys=True, data=True):
        data["cost"] = int(data.get("cost", 0))
        if "weight" not in data:
            data["weight"] = 0.0
    q = int(delta)
    paths: list[list[int]] = []
    base_weight = 0.0
    used_cost = 0
    for _i in range(k):
        found = bounded_cost_shortest_path(R, s, t, q)
        if found is None:
            return LDPResult(False, paths, R, base_weight, used_cost, q, "no bounded-cost augmenting path")
        edge_ids, weight, cost = found
        paths.append(edge_ids)
        base_weight += weight
        used_cost += cost
        q -= cost
        update_restricted_residual(R, edge_ids, s, t)
    return LDPResult(True, paths, R, base_weight, used_cost, q, "ok")
