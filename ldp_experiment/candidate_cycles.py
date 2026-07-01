"""Reverse-edge anchored candidate cycle enumeration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import networkx as nx

from .graph_utils import EPS, edge_by_id, outgoing_edges


@dataclass(frozen=True)
class Cycle:
    edge_ids: tuple[int, ...]
    weight: float
    cost: int
    kind: str


def _cycle_kind(weight: float, cost: int) -> Optional[str]:
    if weight < -EPS and cost > 0:
        return "neg_weight_pos_cost"
    if weight >= -EPS and cost < 0:
        return "nonneg_weight_neg_cost"
    return None


def enumerate_candidate_cycles(
    R: nx.MultiDiGraph,
    exact: bool = True,
    max_cycles_per_anchor: Optional[int] = None,
) -> list[Cycle]:
    """Enumerate useful simple directed cycles using reverse edges as anchors.

    Every useful candidate must contain a reverse edge, so anchors are reverse
    edges. DFS expansion still traverses all directed residual edges because the
    rest of the cycle may use forward, reverse, and split edges.
    """
    by_id = edge_by_id(R)
    anchors = [edge for edge in by_id.values() if edge.is_reverse]
    anchors.sort(key=lambda e: e.eid)
    reverse_rank = {edge.eid: i for i, edge in enumerate(anchors)}
    seen: set[frozenset[int]] = set()
    cycles: list[Cycle] = []

    for anchor in anchors:
        produced = 0
        start = anchor.v
        target = anchor.u

        def dfs(u: Any, visited_nodes: set[Any], path: list[int], weight: float, cost: int) -> None:
            nonlocal produced
            if not exact and max_cycles_per_anchor is not None and produced >= max_cycles_per_anchor:
                return
            for edge in outgoing_edges(R, u):
                if edge.eid == anchor.eid:
                    continue
                if edge.is_reverse and reverse_rank.get(edge.eid, 10**18) < reverse_rank[anchor.eid]:
                    continue
                if edge.v != target and edge.v in visited_nodes:
                    continue
                if edge.v == target:
                    edge_ids = tuple([anchor.eid] + path + [edge.eid])
                    key = frozenset(edge_ids)
                    if key in seen:
                        continue
                    w = anchor.weight + weight + edge.weight
                    c = anchor.cost + cost + edge.cost
                    kind = _cycle_kind(w, c)
                    if kind is None:
                        continue
                    seen.add(key)
                    cycles.append(Cycle(edge_ids=edge_ids, weight=w, cost=c, kind=kind))
                    produced += 1
                    continue
                visited_nodes.add(edge.v)
                path.append(edge.eid)
                dfs(edge.v, visited_nodes, path, weight + edge.weight, cost + edge.cost)
                path.pop()
                visited_nodes.remove(edge.v)

        dfs(start, {start}, [], 0.0, 0)
    return cycles


def validate_candidate_cycles(R: nx.MultiDiGraph, cycles: list[Cycle]) -> None:
    """Raise ValueError if a candidate cycle violates edge-id or classification rules."""
    by_id = edge_by_id(R)
    for cycle in cycles:
        if not cycle.edge_ids:
            raise ValueError("empty cycle")
        nodes = []
        weight = 0.0
        cost = 0
        first = by_id.get(cycle.edge_ids[0])
        if first is None:
            raise ValueError(f"missing eid {cycle.edge_ids[0]}")
        cur = first.u
        nodes.append(cur)
        for eid in cycle.edge_ids:
            edge = by_id.get(eid)
            if edge is None:
                raise ValueError(f"missing eid {eid}")
            if edge.u != cur:
                raise ValueError(f"cycle is not directed at eid {eid}")
            cur = edge.v
            nodes.append(cur)
            weight += edge.weight
            cost += edge.cost
        if cur != nodes[0]:
            raise ValueError("cycle is not closed")
        internal = nodes[:-1]
        if len(internal) != len(set(internal)):
            raise ValueError("cycle repeats a node")
        if abs(weight - cycle.weight) > EPS or cost != cycle.cost:
            raise ValueError("stored weight/cost mismatch")
        if _cycle_kind(weight, cost) != cycle.kind:
            raise ValueError("stored kind mismatch")
