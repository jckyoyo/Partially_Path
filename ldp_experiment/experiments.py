"""Random graph experiment driver and CLI."""

from __future__ import annotations

import argparse
import csv
from typing import Any, Optional

import networkx as nx

from .candidate_cycles import enumerate_candidate_cycles
from .conflict_dp import solve_by_conflict_dp
from .graph_utils import EPS, add_edge_with_attrs
from .ldp_algorithm import run_ldp
from .residual_ilp import solve_candidate_edge_subgraph_ilp, solve_residual_circulation_ilp


def generate_random_digraph(
    n: int,
    m: int,
    weight_low: int = 1,
    weight_high: int = 100,
    seed: Optional[int] = None,
) -> nx.MultiDiGraph:
    """Generate a directed MultiDiGraph with positive integer weights and eids."""
    rng = nx.utils.create_random_state(seed)
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(n))
    attempts = 0
    while G.number_of_edges() < m and attempts < max(10 * m, 100):
        attempts += 1
        u = int(rng.randint(0, n))
        v = int(rng.randint(0, n))
        if u == v:
            continue
        add_edge_with_attrs(G, u, v, weight=int(rng.randint(weight_low, weight_high + 1)), cost=0, desc="random")
    return G


def run_single_experiment(
    G: nx.MultiDiGraph,
    s: Any,
    t: Any,
    k: int,
    delta: int,
    exact_cycle_enum: bool = True,
    gurobi_time_limit: Optional[float] = None,
) -> dict[str, Any]:
    """Run LDP, candidate-cycle DP, full residual ILP, and candidate-edge ILP."""
    ldp = run_ldp(G, s, t, k, delta)
    row: dict[str, Any] = {
        "s": s,
        "t": t,
        "ldp_feasible": ldp.feasible,
        "base_weight": ldp.base_weight,
        "remaining_budget": ldp.remaining_budget,
        "num_residual_nodes": ldp.residual.number_of_nodes(),
        "num_residual_edges": ldp.residual.number_of_edges(),
    }
    if not ldp.feasible:
        row.update({"message": ldp.message})
        return row
    cycles = enumerate_candidate_cycles(ldp.residual, exact=exact_cycle_enum)
    dp = solve_by_conflict_dp(cycles, ldp.remaining_budget)
    full = solve_residual_circulation_ilp(ldp.residual, ldp.remaining_budget, time_limit=gurobi_time_limit)
    cand = solve_candidate_edge_subgraph_ilp(ldp.residual, cycles, ldp.remaining_budget, time_limit=gurobi_time_limit)
    row.update(
        {
            "num_candidates": len(cycles),
            "num_components": dp.num_components,
            "max_component_size": dp.max_component_size,
            "dp_objective": dp.objective,
            "dp_cost": dp.total_cost,
            "dp_improved": dp.improved,
            "dp_time": dp.runtime_sec,
            "num_dp_states": dp.num_dp_states,
            "full_ilp_objective": full.objective,
            "full_ilp_cost": full.total_cost,
            "full_ilp_improved": full.improved,
            "full_ilp_time": full.runtime_sec,
            "full_ilp_status": full.status,
            "cand_ilp_objective": cand.objective,
            "cand_ilp_cost": cand.total_cost,
            "cand_ilp_improved": cand.improved,
            "cand_ilp_time": cand.runtime_sec,
            "cand_ilp_status": cand.status,
            "dp_matches_full_ilp": full.status == "NO_GUROBI" or abs(dp.objective - full.objective) <= EPS,
            "cand_ilp_matches_full_ilp": full.status == "NO_GUROBI" or abs(cand.objective - full.objective) <= EPS,
        }
    )
    return row


CSV_FIELDS = [
    "trial", "n", "m", "k", "delta", "s", "t",
    "ldp_feasible", "base_weight", "remaining_budget",
    "num_residual_nodes", "num_residual_edges",
    "num_candidates", "num_components", "max_component_size",
    "num_dp_states",
    "dp_objective", "dp_cost", "dp_improved", "dp_time",
    "full_ilp_objective", "full_ilp_cost", "full_ilp_improved", "full_ilp_time", "full_ilp_status",
    "cand_ilp_objective", "cand_ilp_cost", "cand_ilp_improved", "cand_ilp_time", "cand_ilp_status",
    "dp_matches_full_ilp", "cand_ilp_matches_full_ilp",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--delta", type=int, required=True)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--gurobi-time-limit", type=float, default=None)
    args = parser.parse_args()
    rows = []
    for trial in range(args.trials):
        G = generate_random_digraph(args.n, args.m, seed=args.seed + trial)
        s, t = 0, args.n - 1
        row = run_single_experiment(G, s, t, args.k, args.delta, gurobi_time_limit=args.gurobi_time_limit)
        row.update({"trial": trial, "n": args.n, "m": args.m, "k": args.k, "delta": args.delta})
        rows.append(row)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
