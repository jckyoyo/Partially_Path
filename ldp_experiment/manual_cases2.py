"""Deterministic hand-built graph cases for debugging the residual postprocess."""

from __future__ import annotations

import argparse
import os
import sys
import time

import networkx as nx

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ldp_experiment.candidate_cycles import enumerate_candidate_cycles, validate_candidate_cycles
from ldp_experiment.conflict_dp import solve_by_conflict_dp
from ldp_experiment.graph_utils import add_edge_with_attrs, edge_by_id
from ldp_experiment.ldp_algorithm import run_ldp
from ldp_experiment.residual_ilp import solve_candidate_edge_subgraph_ilp, solve_residual_circulation_ilp


MANUAL_EDGES = [
    ("s", "a", 1),
    ("a", "b", 1),
    ("b", "c", 1),
    ("c", "d", 1),
    ("d", "e", 1),
    ("e", "t", 1),
    ("s", "f", 1),
    ("f", "g", 1),
    ("g", "h", 1),
    ("h", "c", 1),
    ("c", "j", 1),
    ("j", "k", 1),
    ("k", "l", 1),
    ("l", "t", 1),
    ("s", "m", 1),
    ("m", "g", 1),
    ("g", "n", 1),
    ("n", "o", 1),
    ("o", "p", 1),
    ("p", "q", 1),
    ("q", "r", 20),
    ("r", "v", 1),
    ("v", "u", 1),
    ("u", "t", 1),
    ("q", "e", 1),
    ("e", "r", 2),
    ("b", "d", 4),
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

    total_start = time.perf_counter()
    G = build_manual_graph()
    ldp_start = time.perf_counter()
    ldp = run_ldp(G, "s", "t", args.k, args.delta)
    ldp_time = time.perf_counter() - ldp_start
    print(f"LDP feasible: {ldp.feasible}, message={ldp.message}")
    print(f"base_weight={ldp.base_weight}, used_cost={ldp.used_cost}, remaining_budget={ldp.remaining_budget}")
    print(f"paths(edge ids)={ldp.paths}")
    print(f"LDP runtime={ldp_time:.6f}s")
    if not ldp.feasible:
        print(f"total runtime={time.perf_counter() - total_start:.6f}s")
        return

    edge_map = edge_by_id(ldp.residual)
    enum_start = time.perf_counter()
    cycles = enumerate_candidate_cycles(
        ldp.residual,
        exact=args.max_cycles_per_anchor is None,
        max_cycles_per_anchor=args.max_cycles_per_anchor,
    )
    enum_only_time = time.perf_counter() - enum_start
    validate_start = time.perf_counter()
    validate_candidate_cycles(ldp.residual, cycles)
    validate_time = time.perf_counter() - validate_start
    enum_time = enum_only_time + validate_time
    print(f"num_candidates={len(cycles)}")
    print(f"candidate enumeration runtime={enum_only_time:.6f}s")
    print(f"candidate validation runtime={validate_time:.6f}s")
    for i, cycle in enumerate(cycles):
        print(f"cycle[{i}] kind={cycle.kind}, weight={cycle.weight}, cost={cycle.cost}")
        print(f"  {describe_cycle(edge_map, cycle.edge_ids)}")

    dp_start = time.perf_counter()
    dp = solve_by_conflict_dp(cycles, ldp.remaining_budget)
    dp_wall_time = time.perf_counter() - dp_start
    our_algorithm_time = enum_time + dp_wall_time
    print(f"DP objective={dp.objective}, cost={dp.total_cost}, improved={dp.improved}, selected={len(dp.selected_cycles)}")
    print(f"DP runtime={dp.runtime_sec:.6f}s, wall={dp_wall_time:.6f}s")
    print(f"our postprocess total runtime={our_algorithm_time:.6f}s")

    full_start = time.perf_counter()
    full = solve_residual_circulation_ilp(ldp.residual, ldp.remaining_budget, time_limit=args.gurobi_time_limit)
    full_wall_time = time.perf_counter() - full_start
    cand_start = time.perf_counter()
    cand = solve_candidate_edge_subgraph_ilp(ldp.residual, cycles, ldp.remaining_budget, time_limit=args.gurobi_time_limit)
    cand_wall_time = time.perf_counter() - cand_start
    print(f"full ILP objective={full.objective}, cost={full.total_cost}, improved={full.improved}, status={full.status}")
    print(f"full ILP runtime={full.runtime_sec:.6f}s, wall={full_wall_time:.6f}s")
    print(f"cand ILP objective={cand.objective}, cost={cand.total_cost}, improved={cand.improved}, status={cand.status}")
    print(f"cand ILP runtime={cand.runtime_sec:.6f}s, wall={cand_wall_time:.6f}s")
    print(f"LDP + our postprocess runtime={ldp_time + our_algorithm_time:.6f}s")
    print(f"LDP + full ILP runtime={ldp_time + full_wall_time:.6f}s")
    print(f"LDP + cand ILP runtime={ldp_time + cand_wall_time:.6f}s")
    print(f"total runtime={time.perf_counter() - total_start:.6f}s")


if __name__ == "__main__":
    main()
