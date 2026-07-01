"""Experiment driver for residual candidate-cycle DP vs residual ILP.

This file is intended to replace ``ldp_experiment/experiments.py``.
It keeps the experiment layer thin: LDP builds a residual graph, the
post-processing modules enumerate candidate cycles and solve the conflict-DP,
and the ILP modules provide full/candidate-edge residual circulation baselines.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from typing import Any, Optional

import networkx as nx

# Allow both usages:
#   python -m ldp_experiment.experiments ...
#   python ldp_experiment/experiments.py ...
if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ldp_experiment.candidate_cycles import (  # noqa: E402
    Cycle,
    enumerate_candidate_cycles,
    validate_candidate_cycles,
)
from ldp_experiment.conflict_dp import DPResult, solve_by_conflict_dp  # noqa: E402
from ldp_experiment.graph_utils import EPS, add_edge_with_attrs  # noqa: E402
from ldp_experiment.ldp_algorithm import run_ldp  # noqa: E402
from ldp_experiment.residual_ilp import (  # noqa: E402
    ILPResult,
    solve_candidate_edge_subgraph_ilp,
    solve_residual_circulation_ilp,
)


OPTIMAL_STATUS = "OPTIMAL"
NO_GUROBI_STATUS = "NO_GUROBI"


def _is_optimal(status: str) -> bool:
    """Return True only when the ILP result is certified optimal."""
    return str(status).upper() == OPTIMAL_STATUS


def _match_if_certified(a: float, b: float, status_a: str, status_b: str) -> str:
    """Return CSV-friendly equality flag for two solver objectives.

    Empty string means at least one side is not certified optimal, so the
    comparison should not be interpreted as a correctness failure.
    """
    if not (_is_optimal(status_a) and _is_optimal(status_b)):
        return ""
    return str(abs(a - b) <= EPS)


def _match_dp_vs_ilp(dp_obj: float, ilp_obj: float, ilp_status: str) -> str:
    """Compare DP against an ILP only when the ILP is certified optimal."""
    if not _is_optimal(ilp_status):
        return ""
    return str(abs(dp_obj - ilp_obj) <= EPS)


def _safe_error_message(exc: BaseException) -> str:
    return f"{exc.__class__.__name__}: {exc}"


def generate_random_digraph(
    n: int,
    m: int,
    weight_low: int = 1,
    weight_high: int = 100,
    seed: Optional[int] = None,
) -> nx.MultiDiGraph:
    """Generate a random directed MultiDiGraph with positive integer weights.

    Notes
    -----
    * Parallel edges are allowed, which is consistent with MultiDiGraph and
      avoids impossible requests when ``m > n * (n - 1)``.
    * Edge IDs are assigned incrementally during generation. This avoids the
      expensive pattern of scanning all existing edges to find the next eid.
    * All generated edges start as original forward edges with cost 0.
    """
    if n <= 1:
        raise ValueError("n must be at least 2")
    if m < 0:
        raise ValueError("m must be non-negative")
    if weight_low <= 0 or weight_high < weight_low:
        raise ValueError("weights must be positive and weight_high >= weight_low")

    rng = random.Random(seed)
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(n))

    for eid in range(m):
        while True:
            u = rng.randrange(n)
            v = rng.randrange(n)
            if u != v:
                break
        add_edge_with_attrs(
            G,
            u,
            v,
            eid=eid,
            weight=rng.randint(weight_low, weight_high),
            cost=0,
            is_reverse=False,
            is_split_edge=False,
            original_eid=None,
            desc="random",
        )
    return G


def _count_cycle_kinds(cycles: list[Cycle]) -> tuple[int, int]:
    num_neg_pos = sum(c.kind == "neg_weight_pos_cost" for c in cycles)
    num_nonneg_neg = sum(c.kind == "nonneg_weight_neg_cost" for c in cycles)
    return num_neg_pos, num_nonneg_neg


def _validate_dp_result(dp: DPResult) -> str:
    """Return empty string if the recovered DP solution is internally consistent."""
    try:
        seen_edges: set[int] = set()
        total_cost = 0
        total_weight = 0.0
        for cyc in dp.selected_cycles:
            edge_set = set(cyc.edge_ids)
            if seen_edges & edge_set:
                return "selected cycles share residual edges"
            seen_edges.update(edge_set)
            total_cost += cyc.cost
            total_weight += cyc.weight
        if total_cost != dp.total_cost:
            return f"selected cost {total_cost} != dp.total_cost {dp.total_cost}"
        if abs(total_weight - dp.objective) > EPS:
            return f"selected weight {total_weight} != dp.objective {dp.objective}"
        return ""
    except Exception as exc:  # pragma: no cover - defensive diagnostics
        return _safe_error_message(exc)


def _blank_row() -> dict[str, Any]:
    """Default row with all optional fields present for stable CSV output."""
    return {field: "" for field in CSV_FIELDS}


def run_single_experiment(
    G: nx.MultiDiGraph,
    s: Any,
    t: Any,
    k: int,
    delta: int,
    exact_cycle_enum: bool = True,
    max_cycles_per_anchor: Optional[int] = None,
    max_exact_component_size: int = 25,
    gurobi_time_limit: Optional[float] = None,
    mip_gap: Optional[float] = None,
    run_ilp: bool = True,
    validate_cycles: bool = True,
) -> dict[str, Any]:
    """Run one full experiment on a supplied NetworkX MultiDiGraph.

    The returned dict is CSV-ready. If a phase fails, the row records the phase
    and error message instead of crashing the whole batch.
    """
    row: dict[str, Any] = _blank_row()
    row.update(
        {
            "s": s,
            "t": t,
            "k": k,
            "delta": delta,
            "cycle_enum_exact": exact_cycle_enum,
            "max_cycles_per_anchor": max_cycles_per_anchor if max_cycles_per_anchor is not None else "",
        }
    )

    # 1. LDP / residual graph construction.
    try:
        ldp = run_ldp(G, s, t, k, delta)
    except Exception as exc:
        row.update({"ldp_feasible": False, "ldp_message": _safe_error_message(exc)})
        return row

    row.update(
        {
            "ldp_feasible": ldp.feasible,
            "ldp_message": ldp.message,
            "base_weight": ldp.base_weight,
            "used_cost": ldp.used_cost,
            "remaining_budget": ldp.remaining_budget,
            "path_count": len(ldp.paths),
            "num_residual_nodes": ldp.residual.number_of_nodes(),
            "num_residual_edges": ldp.residual.number_of_edges(),
        }
    )
    if not ldp.feasible:
        return row

    B = ldp.remaining_budget

    # 2. Candidate-cycle enumeration.
    try:
        cycles = enumerate_candidate_cycles(
            ldp.residual,
            exact=exact_cycle_enum,
            max_cycles_per_anchor=max_cycles_per_anchor,
        )
        if validate_cycles:
            validate_candidate_cycles(ldp.residual, cycles)
    except Exception as exc:
        row.update({"cycle_enum_error": _safe_error_message(exc)})
        return row

    num_neg_pos, num_nonneg_neg = _count_cycle_kinds(cycles)
    row.update(
        {
            "num_candidates": len(cycles),
            "num_neg_weight_pos_cost": num_neg_pos,
            "num_nonneg_weight_neg_cost": num_nonneg_neg,
        }
    )

    # 3. Conflict-DP.
    try:
        dp = solve_by_conflict_dp(cycles, B, max_exact_component_size=max_exact_component_size)
        dp_validation_error = _validate_dp_result(dp)
        row.update(
            {
                "num_components": dp.num_components,
                "max_component_size": dp.max_component_size,
                "num_dp_states": dp.num_dp_states,
                "dp_objective": dp.objective,
                "dp_cost": dp.total_cost,
                "dp_improved": dp.improved,
                "dp_time": dp.runtime_sec,
                "dp_selected_count": len(dp.selected_cycles),
                "dp_validation_error": dp_validation_error,
            }
        )
    except Exception as exc:
        row.update({"dp_error": _safe_error_message(exc)})
        return row

    # 4. Optional ILP baselines.
    if run_ilp:
        try:
            full = solve_residual_circulation_ilp(
                ldp.residual,
                B,
                time_limit=gurobi_time_limit,
                mip_gap=mip_gap,
            )
        except Exception as exc:
            full = ILPResult(0.0, 0, [], False, f"ERROR:{_safe_error_message(exc)}", 0.0)
        try:
            cand = solve_candidate_edge_subgraph_ilp(
                ldp.residual,
                cycles,
                B,
                time_limit=gurobi_time_limit,
                mip_gap=mip_gap,
            )
        except Exception as exc:
            cand = ILPResult(0.0, 0, [], False, f"ERROR:{_safe_error_message(exc)}", 0.0)
    else:
        full = ILPResult(0.0, 0, [], False, "SKIPPED", 0.0)
        cand = ILPResult(0.0, 0, [], False, "SKIPPED", 0.0)

    row.update(
        {
            "full_ilp_objective": full.objective,
            "full_ilp_cost": full.total_cost,
            "full_ilp_improved": full.improved,
            "full_ilp_time": full.runtime_sec,
            "full_ilp_status": full.status,
            "full_ilp_selected_edges": len(full.selected_edge_ids),
            "cand_ilp_objective": cand.objective,
            "cand_ilp_cost": cand.total_cost,
            "cand_ilp_improved": cand.improved,
            "cand_ilp_time": cand.runtime_sec,
            "cand_ilp_status": cand.status,
            "cand_ilp_selected_edges": len(cand.selected_edge_ids),
            # Only compare against certified optimal ILP results.
            "dp_matches_full_ilp": _match_dp_vs_ilp(dp.objective, full.objective, full.status),
            "dp_matches_cand_ilp": _match_dp_vs_ilp(dp.objective, cand.objective, cand.status),
            "cand_ilp_matches_full_ilp": _match_if_certified(cand.objective, full.objective, cand.status, full.status),
        }
    )
    return row


CSV_FIELDS = [
    "trial",
    "seed",
    "n",
    "m",
    "k",
    "delta",
    "s",
    "t",
    "cycle_enum_exact",
    "max_cycles_per_anchor",
    "ldp_feasible",
    "ldp_message",
    "base_weight",
    "used_cost",
    "remaining_budget",
    "path_count",
    "num_residual_nodes",
    "num_residual_edges",
    "num_candidates",
    "num_neg_weight_pos_cost",
    "num_nonneg_weight_neg_cost",
    "num_components",
    "max_component_size",
    "num_dp_states",
    "dp_objective",
    "dp_cost",
    "dp_improved",
    "dp_time",
    "dp_selected_count",
    "dp_validation_error",
    "full_ilp_objective",
    "full_ilp_cost",
    "full_ilp_improved",
    "full_ilp_time",
    "full_ilp_status",
    "full_ilp_selected_edges",
    "cand_ilp_objective",
    "cand_ilp_cost",
    "cand_ilp_improved",
    "cand_ilp_time",
    "cand_ilp_status",
    "cand_ilp_selected_edges",
    "dp_matches_full_ilp",
    "dp_matches_cand_ilp",
    "cand_ilp_matches_full_ilp",
    "cycle_enum_error",
    "dp_error",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LDP residual post-processing experiments.")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--delta", type=int, required=True)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--target", type=int, default=None)
    parser.add_argument("--weight-low", type=int, default=1)
    parser.add_argument("--weight-high", type=int, default=100)
    parser.add_argument("--gurobi-time-limit", type=float, default=None)
    parser.add_argument("--mip-gap", type=float, default=None)
    parser.add_argument("--skip-ilp", action="store_true", help="Skip both full and candidate-edge ILP baselines.")
    parser.add_argument("--heuristic-cycle-enum", action="store_true")
    parser.add_argument("--max-cycles-per-anchor", type=int, default=None)
    parser.add_argument("--max-exact-component-size", type=int, default=25)
    parser.add_argument("--no-validate-cycles", action="store_true")
    parser.add_argument("--print-progress", action="store_true")
    args = parser.parse_args()

    if args.heuristic_cycle_enum and args.max_cycles_per_anchor is None:
        parser.error("--heuristic-cycle-enum requires --max-cycles-per-anchor; otherwise enumeration is not truncated")

    target = args.n - 1 if args.target is None else args.target
    rows: list[dict[str, Any]] = []

    for trial in range(args.trials):
        trial_seed = args.seed + trial
        if args.print_progress:
            print(f"trial {trial + 1}/{args.trials}, seed={trial_seed}", flush=True)
        G = generate_random_digraph(
            args.n,
            args.m,
            weight_low=args.weight_low,
            weight_high=args.weight_high,
            seed=trial_seed,
        )
        row = run_single_experiment(
            G,
            args.source,
            target,
            args.k,
            args.delta,
            exact_cycle_enum=not args.heuristic_cycle_enum,
            max_cycles_per_anchor=args.max_cycles_per_anchor,
            max_exact_component_size=args.max_exact_component_size,
            gurobi_time_limit=args.gurobi_time_limit,
            mip_gap=args.mip_gap,
            run_ilp=not args.skip_ilp,
            validate_cycles=not args.no_validate_cycles,
        )
        row.update(
            {
                "trial": trial,
                "seed": trial_seed,
                "n": args.n,
                "m": args.m,
                "k": args.k,
                "delta": args.delta,
            }
        )
        rows.append(row)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    if args.print_progress:
        print(f"wrote {len(rows)} rows to {args.out}", flush=True)


if __name__ == "__main__":
    main()
