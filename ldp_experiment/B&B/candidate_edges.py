"""Candidate key-edge extraction for residual post-processing.

The returned edge IDs are only branching hints for exact algorithms. They do
not define a restricted optimization model, and non-candidate residual edges
must remain available to any downstream full-graph solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx

from ldp_experiment.graph_utils import EPS, edge_by_id


@dataclass(frozen=True)
class CandidateEdgeStats:
    num_negative_weight_edges: int
    num_negative_cost_edges: int
    num_after_positive_in_prune: int
    num_after_scc_prune: int
    num_candidate_edges: int


def _has_positive_in_edge(R: nx.MultiDiGraph, node: Any) -> bool:
    for _u, _v, _key, data in R.in_edges(node, keys=True, data=True):
        if float(data.get("weight", 0.0)) > EPS:
            return True
    return False


def extract_candidate_key_edges_with_stats(R: nx.MultiDiGraph) -> tuple[set[int], CandidateEdgeStats]:
    """Return key residual edge IDs and pruning statistics.

    Negative-cost edges are always useful branching candidates when they can
    lie on some directed cycle. Negative-weight edges are kept only when the
    tail has a positive-weight incoming edge, matching the local anchor filter
    used by candidate-cycle enumeration. SCC pruning then removes edges whose
    endpoints cannot be in the same directed circulation.
    """
    by_id = edge_by_id(R)
    scc_id: dict[Any, int] = {}
    for sid, comp in enumerate(nx.strongly_connected_components(R)):
        for node in comp:
            scc_id[node] = sid

    negative_weight = {e.eid for e in by_id.values() if e.weight < -EPS}
    negative_cost = {e.eid for e in by_id.values() if e.cost < 0}
    after_positive_in = {
        eid
        for eid in negative_weight
        if _has_positive_in_edge(R, by_id[eid].u)
    } | negative_cost
    candidates = {
        eid
        for eid in after_positive_in
        if scc_id.get(by_id[eid].u) == scc_id.get(by_id[eid].v)
    }
    stats = CandidateEdgeStats(
        num_negative_weight_edges=len(negative_weight),
        num_negative_cost_edges=len(negative_cost),
        num_after_positive_in_prune=len(after_positive_in),
        num_after_scc_prune=len(candidates),
        num_candidate_edges=len(candidates),
    )
    return candidates, stats


def extract_candidate_key_edges(R: nx.MultiDiGraph) -> set[int]:
    """Return key edge IDs for candidate-edge-guided branching."""
    candidates, _stats = extract_candidate_key_edges_with_stats(R)
    return candidates
