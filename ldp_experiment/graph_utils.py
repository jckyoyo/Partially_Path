"""Utilities for MultiDiGraph edge-id based algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import networkx as nx


EPS = 1e-9


@dataclass(frozen=True)
class EdgeRef:
    u: Any
    v: Any
    key: Any
    eid: int
    weight: float
    cost: int
    is_reverse: bool
    is_split_edge: bool
    original_eid: Optional[int]
    desc: str


def next_eid(G: nx.MultiDiGraph) -> int:
    """Return the next unused integer edge id."""
    eids = [data.get("eid") for _, _, _, data in G.edges(keys=True, data=True)]
    int_eids = [int(eid) for eid in eids if eid is not None]
    return max(int_eids, default=-1) + 1


def add_edge_with_attrs(
    G: nx.MultiDiGraph,
    u: Any,
    v: Any,
    *,
    eid: Optional[int] = None,
    weight: float = 0.0,
    cost: int = 0,
    is_reverse: bool = False,
    is_split_edge: bool = False,
    original_eid: Optional[int] = None,
    desc: str = "",
) -> int:
    """Add one residual/original edge and return its unique eid."""
    if eid is None:
        eid = next_eid(G)
    G.add_edge(
        u,
        v,
        key=eid,
        eid=eid,
        weight=float(weight),
        cost=int(cost),
        is_reverse=bool(is_reverse),
        is_split_edge=bool(is_split_edge),
        original_eid=original_eid,
        desc=desc,
    )
    return eid


def ensure_edge_ids(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Return a copy with required attributes and unique eids on every edge."""
    H = nx.MultiDiGraph()
    H.add_nodes_from(G.nodes(data=True))
    used: set[int] = set()
    nxt = 0
    for u, v, _key, data in G.edges(keys=True, data=True):
        eid = data.get("eid")
        if eid is None or eid in used:
            while nxt in used:
                nxt += 1
            eid = nxt
        used.add(int(eid))
        add_edge_with_attrs(
            H,
            u,
            v,
            eid=int(eid),
            weight=float(data.get("weight", 0.0)),
            cost=int(data.get("cost", 0)),
            is_reverse=bool(data.get("is_reverse", False)),
            is_split_edge=bool(data.get("is_split_edge", False)),
            original_eid=data.get("original_eid"),
            desc=str(data.get("desc", "")),
        )
    return H


def edge_by_id(G: nx.MultiDiGraph) -> dict[int, EdgeRef]:
    """Build an eid -> EdgeRef map."""
    out: dict[int, EdgeRef] = {}
    for u, v, key, data in G.edges(keys=True, data=True):
        eid = int(data["eid"])
        if eid in out:
            raise ValueError(f"duplicate eid {eid}")
        out[eid] = EdgeRef(
            u=u,
            v=v,
            key=key,
            eid=eid,
            weight=float(data.get("weight", 0.0)),
            cost=int(data.get("cost", 0)),
            is_reverse=bool(data.get("is_reverse", False)),
            is_split_edge=bool(data.get("is_split_edge", False)),
            original_eid=data.get("original_eid"),
            desc=str(data.get("desc", "")),
        )
    return out


def edge_ids_on_path_weight_cost(G: nx.MultiDiGraph, edge_ids: Iterable[int]) -> tuple[float, int]:
    """Return total weight and cost for a sequence of edge ids."""
    by_id = edge_by_id(G)
    weight = 0.0
    cost = 0
    for eid in edge_ids:
        edge = by_id[eid]
        weight += edge.weight
        cost += edge.cost
    return weight, cost


def remove_edge_by_id(G: nx.MultiDiGraph, eid: int) -> EdgeRef:
    """Remove and return the edge with the given eid."""
    edge = edge_by_id(G)[eid]
    G.remove_edge(edge.u, edge.v, edge.key)
    return edge


def outgoing_edges(G: nx.MultiDiGraph, u: Any) -> list[EdgeRef]:
    """Return outgoing edges as EdgeRef objects."""
    out = []
    for _, v, key, data in G.out_edges(u, keys=True, data=True):
        out.append(
            EdgeRef(
                u=u,
                v=v,
                key=key,
                eid=int(data["eid"]),
                weight=float(data.get("weight", 0.0)),
                cost=int(data.get("cost", 0)),
                is_reverse=bool(data.get("is_reverse", False)),
                is_split_edge=bool(data.get("is_split_edge", False)),
                original_eid=data.get("original_eid"),
                desc=str(data.get("desc", "")),
            )
        )
    return out
