"""Deterministic hand-built graph cases for debugging the residual postprocess."""

from __future__ import annotations

import argparse
import os
import sys

import networkx as nx

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ldp_experiment.candidate_cycles import enumerate_candidate_cycles, validate_candidate_cycles
from ldp_experiment.conflict_dp import solve_by_conflict_dp
from ldp_experiment.graph_utils import add_edge_with_attrs, edge_by_id
from ldp_experiment.ldp_algorithm import run_ldp
from ldp_experiment.residual_ilp import solve_candidate_edge_subgraph_ilp, solve_residual_circulation_ilp


MANUAL_EDGES = [
    ("s", "p", 1),
    ("p", "x", 1),
    ("x", "u", 1),
    ("u", "a", 1),
    ("a", "q", 1),
    ("q", "r", 20),
    ("r", "t", 1),
    ("s", "z", 1),
    ("z", "x", 1),
    ("x", "m", 1),
    ("m", "a", 1),
    ("a", "n", 1),
    ("n", "t", 1),
    ("s", "l", 1),
    ("l", "b", 1),
    ("b", "h", 1),
    ("h", "t", 1),
    ("q", "b", 1),
    ("b", "r", 2),
    ("m", "n", 3),
]


def build_manual_graph() -> nx.MultiDiGraph:
    """Build the hand-written graph supplied for algorithm debugging."""
    G = nx.MultiDiGraph()
    for u, v, weight in MANUAL_EDGES:
        add_edge_with_attrs(G, u, v, weight=weight, cost=0, desc="manual")
    return G


def describe_cycle(edge_map, edge_ids: tuple[int, ...]) -> str:
    parts = []
    for eid in edge_ids:
        edge = edge_map[eid]
        marker = "R" if edge.is_reverse else "F"
        parts.append(f"{eid}:{edge.u}->{edge.v}:w={edge.weight}:c={edge.cost}:{marker}")
    return " | ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed manual graph through LDP, candidate cycles, DP, and ILP.")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--delta", type=int, default=2)
    parser.add_argument("--max-cycles-per-anchor", type=int, default=None)
    parser.add_argument("--gurobi-time-limit", type=float, default=5.0)
    args = parser.parse_args()

    G = build_manual_graph()
    ldp = run_ldp(G, "s", "t", args.k, args.delta)
    print(f"LDP feasible: {ldp.feasible}, message={ldp.message}")
    print(f"base_weight={ldp.base_weight}, used_cost={ldp.used_cost}, remaining_budget={ldp.remaining_budget}")
    print(f"paths(edge ids)={ldp.paths}")
    if not ldp.feasible:
        return

    edge_map = edge_by_id(ldp.residual)
    cycles = enumerate_candidate_cycles(
        ldp.residual,
        exact=args.max_cycles_per_anchor is None,
        max_cycles_per_anchor=args.max_cycles_per_anchor,
    )
    validate_candidate_cycles(ldp.residual, cycles)
    print(f"num_candidates={len(cycles)}")
    for i, cycle in enumerate(cycles):
        print(f"cycle[{i}] kind={cycle.kind}, weight={cycle.weight}, cost={cycle.cost}")
        print(f"  {describe_cycle(edge_map, cycle.edge_ids)}")

    dp = solve_by_conflict_dp(cycles, ldp.remaining_budget)
    print(f"DP objective={dp.objective}, cost={dp.total_cost}, improved={dp.improved}, selected={len(dp.selected_cycles)}")

    full = solve_residual_circulation_ilp(ldp.residual, ldp.remaining_budget, time_limit=args.gurobi_time_limit)
    cand = solve_candidate_edge_subgraph_ilp(ldp.residual, cycles, ldp.remaining_budget, time_limit=args.gurobi_time_limit)
    print(f"full ILP objective={full.objective}, cost={full.total_cost}, improved={full.improved}, status={full.status}")
    print(f"cand ILP objective={cand.objective}, cost={cand.total_cost}, improved={cand.improved}, status={cand.status}")


if __name__ == "__main__":
    main()
