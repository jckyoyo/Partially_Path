"""Deterministic hand-built graph cases for debugging the residual postprocess."""

from __future__ import annotations

import argparse
import os
import sys
import time

import networkx as nx

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ldp_experiment.candidate_cycles import enumerate_candidate_cycles, validate_candidate_cycles
from ldp_experiment.candidate_edges import extract_candidate_key_edges
from ldp_experiment.conflict_dp import solve_by_conflict_dp
from ldp_experiment.graph_utils import add_edge_with_attrs, edge_by_id
from ldp_experiment.lagrangian_bnb import solve_candidate_edge_lagrangian_bnb
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
    parser.add_argument("--run-bnb", action="store_true")
    parser.add_argument("--bnb-max-nodes", type=int, default=None)
    parser.add_argument("--bnb-lagrangian-iters", type=int, default=30)
    parser.add_argument("--bnb-time-limit", type=float, default=None)
    parser.add_argument("--bnb-tail-time-limit", type=float, default=None)
    parser.add_argument("--bnb-dynamic-scc-pruning", action="store_true")
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

    print(f"DP runtime={dp.runtime_sec:.6f}s, wall={dp_wall_time:.6f}s")
    print(f"our postprocess total runtime={our_algorithm_time:.6f}s")

    full_start = time.perf_counter()
    full = solve_residual_circulation_ilp(ldp.residual, ldp.remaining_budget, time_limit=args.gurobi_time_limit)
    full_wall_time = time.perf_counter() - full_start
    cand_start = time.perf_counter()
    cand = solve_candidate_edge_subgraph_ilp(ldp.residual, cycles, ldp.remaining_budget, time_limit=args.gurobi_time_limit)
    cand_wall_time = time.perf_counter() - cand_start
    candidate_eids = extract_candidate_key_edges(ldp.residual)
    if args.run_bnb:
        bnb_start = time.perf_counter()
        bnb = solve_candidate_edge_lagrangian_bnb(
            ldp.residual,
            ldp.remaining_budget,
            candidate_eids=candidate_eids,
            max_nodes=args.bnb_max_nodes,
            max_lagrangian_iters=args.bnb_lagrangian_iters,
            time_limit=args.bnb_time_limit,
            exact_tail_time_limit=args.bnb_tail_time_limit,
            enable_dynamic_scc_pruning=args.bnb_dynamic_scc_pruning,
        )
        bnb_wall_time = time.perf_counter() - bnb_start
    else:
        bnb = None
        bnb_wall_time = 0.0

    print("-----------------测试总运行时间--------------------")
    print(f"total runtime={time.perf_counter() - total_start:.6f}s")

    print("-----------------ldp运行时间--------------------")
    print(f"LDP runtime={ldp_time:.6f}s")
    print("-----------------our algorithm--------------------")
    print(f"our postprocess total runtime={our_algorithm_time:.6f}s")
    print(f"LDP + our postprocess runtime={ldp_time + our_algorithm_time:.6f}s")
    print("-----------------ILP algorithm--------------------")
    print(f"full ILP runtime={full.runtime_sec:.6f}s, wall={full_wall_time:.6f}s")
    print(f"cand ILP runtime={cand.runtime_sec:.6f}s, wall={cand_wall_time:.6f}s")
    print(f"candidate key edges={len(candidate_eids)}")
    if bnb is not None:
        print(f"B&B runtime={bnb.runtime_sec:.6f}s, wall={bnb_wall_time:.6f}s")
    print(f"LDP + full ILP runtime={ldp_time + full_wall_time:.6f}s")
    print(f"LDP + cand ILP runtime={ldp_time + cand_wall_time:.6f}s")
    if bnb is not None:
        print(f"LDP + B&B runtime={ldp_time + bnb_wall_time:.6f}s")

    print("-----------------输出值对比--------------------")
    print(f"DP objective={dp.objective}, cost={dp.total_cost}, improved={dp.improved}, selected={len(dp.selected_cycles)}")
    print(f"full ILP objective={full.objective}, cost={full.total_cost}, improved={full.improved}, status={full.status}")
    print(f"cand ILP objective={cand.objective}, cost={cand.total_cost}, improved={cand.improved}, status={cand.status}")
    if bnb is not None:
        print(
            f"B&B objective={bnb.objective}, cost={bnb.total_cost}, improved={bnb.improved}, "
            f"status={bnb.status}, nodes={bnb.num_nodes}, pruned_bound={bnb.num_pruned_by_bound}, "
            f"pruned_scc={bnb.num_pruned_by_scc}, tail_calls={bnb.num_exact_tail_calls}, best_lb={bnb.best_lb}"
        )
if __name__ == "__main__":
    main()
