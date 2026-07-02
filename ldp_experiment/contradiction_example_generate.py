"""Generators for planted LDP budget-swap counterexamples.

The legacy public function ``generate_k_path_swap_counterexample`` is kept
compatible: it still returns ``(edges, k, delta, info)``. The dataclass wrapper
below is an interface convenience only and does not change generation logic.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import time
from typing import Any

import networkx as nx

from ldp_experiment.candidate_cycles import enumerate_candidate_cycles, validate_candidate_cycles
from ldp_experiment.conflict_dp import solve_by_conflict_dp
from ldp_experiment.graph_utils import add_edge_with_attrs, edge_by_id
from ldp_experiment.ldp_algorithm import run_ldp
from ldp_experiment.residual_ilp import solve_candidate_edge_subgraph_ilp, solve_residual_circulation_ilp


WeightedEdge = tuple[str, str, int]


@dataclass(frozen=True)
class CounterexampleInstance:
    """Structured view of a generated counterexample instance."""

    edges: list[WeightedEdge]
    k: int
    delta: int
    info: dict[str, Any]

    def as_legacy_tuple(self) -> tuple[list[WeightedEdge], int, int, dict[str, Any]]:
        """Return the historical ``(edges, k, delta, info)`` tuple."""
        return self.edges, self.k, self.delta, self.info


def generate_k_path_swap_counterexample(
    k: int,
    slots_per_pair: int = 3,
    release_alts: int = 2,
    use_alts: int = 2,
    alpha: int = 2,
    M: int = 40,
) -> tuple[list[WeightedEdge], int, int, dict[str, Any]]:
    """
    Generate a directed weighted graph family where an LDP-style sequential
    algorithm is likely to consume budget on bad common nodes x_{p,i},
    while the optimal solution releases those nodes and uses good common
    nodes y_{p,i}.

    Parameters
    ----------
    k : int
        Number of required link-disjoint s-t paths. Must be at least 3.

    slots_per_pair : int
        Number of budget-swap slots in each blocker-saver pair.
        Larger value gives more candidate cycles.

    release_alts : int
        Number of alternative bypasses around each bad node x_{p,i}.
        This increases the number of nonnegative-weight negative-cost cycles.

    use_alts : int
        Number of cheap alternatives through each good node y_{p,i}.
        This increases the number of negative-weight positive-cost cycles.

    alpha : int
        Extra weight paid when P0 bypasses a bad node x_{p,i}.
        The release cycle roughly has weight +alpha and cost -1.

    M : int
        Weight of each expensive direct edge in saver paths.
        Should be much larger than the cheap alternative weight plus alpha.

    Returns
    -------
    edges : list[tuple[str, str, int]]
        Directed weighted edges.

    k : int
        Number of required paths.

    delta : int
        Common-node budget.

    info : dict
        Intended LDP pattern, intended optimal pattern, and theoretical data.
    """

    if k < 3:
        raise ValueError("k must be at least 3.")

    edges: list[WeightedEdge] = []

    def add(u, v, w):
        edges.append((u, v, w))

    g = (k - 1) // 2
    m = slots_per_pair
    total_slots = g * m

    # Common-node budget.
    # LDP spends it on all x_{p,i}.
    # Optimal spends it on all y_{p,i}.
    delta = total_slots

    # ------------------------------------------------------------
    # 1. Build base path P0.
    #
    # P0_bad goes through x_{p,i}.
    # P0_good bypasses x_{p,i} but still goes through y_{p,i}.
    # ------------------------------------------------------------

    P0_bad = ["s"]
    P0_good = ["s"]

    prev = "s"

    for p in range(1, g + 1):
        for i in range(1, m + 1):
            L = f"L_{p}_{i}"
            x = f"x_{p}_{i}"
            R = f"R_{p}_{i}"
            y = f"y_{p}_{i}"
            N = f"N_{p}_{i}"

            # Enter this slot.
            add(prev, L, 1)

            # Bad route through x: L -> x -> R.
            add(L, x, 1)
            add(x, R, 1)

            # Multiple bypass alternatives around x.
            # Each gives a release candidate cycle.
            for a in range(1, release_alts + 1):
                bnode = f"bypass_{p}_{i}_{a}"
                # Total bypass weight = 2 + alpha + (a-1).
                add(L, bnode, 1)
                add(bnode, R, 1 + alpha + (a - 1))

            # Both bad and good base paths pass y.
            add(R, y, 1)
            add(y, N, 1)

            P0_bad.extend([L, x, R, y, N])
            P0_good.extend([L, R, y, N])

            prev = N

    add(prev, "t", 1)
    P0_bad.append("t")
    P0_good.append("t")

    P0_bad_weight = 5 * total_slots + 1
    P0_good_weight = P0_bad_weight + alpha * total_slots

    # ------------------------------------------------------------
    # Choose path weights to enforce intended LDP order:
    #
    # P0_bad first.
    # Then all blockers B_p.
    # Then all savers D_p, but forced to use expensive edges
    # because the budget is already consumed by x nodes.
    # ------------------------------------------------------------

    blocker_gate = P0_bad_weight + 10
    blocker_unit = 1
    blocker_end = 1

    blocker_weight = blocker_gate + 2 * m * blocker_unit + blocker_end

    saver_gate = blocker_weight + 10
    cheap_alt_base = 4
    saver_end = 1

    saver_cheap_weight = saver_gate + m * cheap_alt_base + saver_end
    saver_expensive_weight = saver_gate + m * M + saver_end

    filler_gate = saver_expensive_weight + 1000

    blocker_paths = []
    saver_exp_paths = []
    saver_cheap_paths = []

    # ------------------------------------------------------------
    # 2. Build blocker paths B_p.
    #
    # B_p passes all x_{p,i}, making them common with P0_bad.
    # ------------------------------------------------------------

    for p in range(1, g + 1):
        path = ["s", f"B_{p}_0"]
        add("s", f"B_{p}_0", blocker_gate)

        prev = f"B_{p}_0"

        for i in range(1, m + 1):
            x = f"x_{p}_{i}"
            nxt = f"B_{p}_{i}"

            add(prev, x, blocker_unit)
            add(x, nxt, blocker_unit)

            path.extend([x, nxt])
            prev = nxt

        add(prev, "t", blocker_end)
        path.append("t")

        blocker_paths.append(path)

    # ------------------------------------------------------------
    # 3. Build saver paths D_p.
    #
    # D_p has two choices at each slot:
    #   expensive direct edge: prev -> next
    #   cheap alternative through y_{p,i}
    #
    # The cheap alternative creates a positive-cost negative-weight
    # candidate cycle after LDP chooses the expensive edge.
    # ------------------------------------------------------------

    for p in range(1, g + 1):
        exp_path = ["s", f"D_{p}_0"]
        cheap_path = ["s", f"D_{p}_0"]

        add("s", f"D_{p}_0", saver_gate)
        prev = f"D_{p}_0"

        for i in range(1, m + 1):
            y = f"y_{p}_{i}"
            nxt = f"D_{p}_{i}"

            # Expensive direct edge.
            add(prev, nxt, M)

            # Multiple cheap alternatives through y.
            # Each gives a use candidate cycle.
            for b in range(1, use_alts + 1):
                in_node = f"use_in_{p}_{i}_{b}"
                out_node = f"use_out_{p}_{i}_{b}"

                # Total cheap alternative weight = 4 + (b-1).
                add(prev, in_node, 1)
                add(in_node, y, 1 + (b - 1))
                add(y, out_node, 1)
                add(out_node, nxt, 1)

            exp_path.append(nxt)
            cheap_path.extend([y, nxt])

            prev = nxt

        add(prev, "t", saver_end)
        exp_path.append("t")
        cheap_path.append("t")

        saver_exp_paths.append(exp_path)
        saver_cheap_paths.append(cheap_path)

    # ------------------------------------------------------------
    # 4. Add filler paths if k is even.
    #
    # These paths are disjoint and very expensive, so they do not
    # participate in the counterexample structure.
    # ------------------------------------------------------------

    filler_paths = []
    used_paths = 1 + 2 * g
    filler_count = k - used_paths

    for h in range(1, filler_count + 1):
        path = ["s", f"F_{h}_0", f"F_{h}_1", "t"]
        add("s", f"F_{h}_0", filler_gate + 10 * h)
        add(f"F_{h}_0", f"F_{h}_1", 1)
        add(f"F_{h}_1", "t", 1)
        filler_paths.append(path)

    # ------------------------------------------------------------
    # 5. Theoretical comparison.
    # ------------------------------------------------------------

    greedy_weight = (
        P0_bad_weight
        + g * blocker_weight
        + g * saver_expensive_weight
        + sum(filler_gate + 10 * h + 2 for h in range(1, filler_count + 1))
    )

    optimal_pattern_weight = (
        P0_good_weight
        + g * blocker_weight
        + g * saver_cheap_weight
        + sum(filler_gate + 10 * h + 2 for h in range(1, filler_count + 1))
    )

    # Per slot:
    # release x costs about alpha,
    # use y saves about M - cheap_alt_base.
    per_slot_improvement = M - cheap_alt_base - alpha

    info = {
        "g_pairs": g,
        "slots_per_pair": m,
        "total_slots": total_slots,
        "delta": delta,
        "P0_bad": P0_bad,
        "P0_good": P0_good,
        "blocker_paths": blocker_paths,
        "saver_expensive_paths": saver_exp_paths,
        "saver_cheap_paths": saver_cheap_paths,
        "filler_paths": filler_paths,
        "P0_bad_weight": P0_bad_weight,
        "P0_good_weight": P0_good_weight,
        "blocker_weight_each": blocker_weight,
        "saver_expensive_weight_each": saver_expensive_weight,
        "saver_cheap_weight_each": saver_cheap_weight,
        "greedy_pattern_weight": greedy_weight,
        "optimal_pattern_weight": optimal_pattern_weight,
        "expected_improvement": greedy_weight - optimal_pattern_weight,
        "per_slot_improvement": per_slot_improvement,
        "candidate_release_cycles_at_least": total_slots * release_alts,
        "candidate_use_cycles_at_least": total_slots * use_alts,
        "candidate_cycles_at_least": total_slots * (release_alts + use_alts),
        "bad_common_nodes": [f"x_{p}_{i}" for p in range(1, g + 1) for i in range(1, m + 1)],
        "good_common_nodes": [f"y_{p}_{i}" for p in range(1, g + 1) for i in range(1, m + 1)],
    }

    return edges, k, delta, info


def generate_counterexample_instance(
    k: int,
    slots_per_pair: int = 3,
    release_alts: int = 2,
    use_alts: int = 2,
    alpha: int = 2,
    M: int = 40,
) -> CounterexampleInstance:
    """Generate the same instance as a dataclass instead of a raw tuple."""
    edges, k_out, delta, info = generate_k_path_swap_counterexample(
        k=k,
        slots_per_pair=slots_per_pair,
        release_alts=release_alts,
        use_alts=use_alts,
        alpha=alpha,
        M=M,
    )
    return CounterexampleInstance(edges=edges, k=k_out, delta=delta, info=info)


def build_counterexample_graph(edges: list[WeightedEdge]) -> nx.MultiDiGraph:
    """Convert weighted edge tuples into the project's MultiDiGraph format."""
    G = nx.MultiDiGraph()
    for u, v, weight in edges:
        add_edge_with_attrs(G, u, v, weight=weight, cost=0, desc="contradiction")
    return G


def describe_cycle(edge_map, edge_ids: tuple[int, ...]) -> str:
    """Return a compact edge-by-edge cycle description for debugging."""
    parts = []
    for eid in edge_ids:
        edge = edge_map[eid]
        marker = "R" if edge.is_reverse else "F"
        parts.append(f"{eid}:{edge.u}->{edge.v}:w={edge.weight}:c={edge.cost}:{marker}")
    return " | ".join(parts)


def run_counterexample_experiment(
    *,
    k: int,
    slots_per_pair: int = 3,
    release_alts: int = 2,
    use_alts: int = 2,
    alpha: int = 2,
    M: int = 40,
    max_cycles_per_anchor: int | None = None,
    gurobi_time_limit: float | None = 5.0,
    run_ilp: bool = True,
) -> dict[str, Any]:
    """Generate a counterexample and run the current LDP/postprocess pipeline."""
    total_start = time.perf_counter()
    instance = generate_counterexample_instance(
        k=k,
        slots_per_pair=slots_per_pair,
        release_alts=release_alts,
        use_alts=use_alts,
        alpha=alpha,
        M=M,
    )
    G = build_counterexample_graph(instance.edges)

    ldp_start = time.perf_counter()
    ldp = run_ldp(G, "s", "t", instance.k, instance.delta)
    ldp_time = time.perf_counter() - ldp_start

    result: dict[str, Any] = {
        "instance": instance,
        "graph": G,
        "ldp": ldp,
        "ldp_time": ldp_time,
        "total_runtime": None,
    }
    if not ldp.feasible:
        result["total_runtime"] = time.perf_counter() - total_start
        return result

    enum_start = time.perf_counter()
    cycles = enumerate_candidate_cycles(
        ldp.residual,
        exact=max_cycles_per_anchor is None,
        max_cycles_per_anchor=max_cycles_per_anchor,
    )
    enum_time = time.perf_counter() - enum_start

    validate_start = time.perf_counter()
    validate_candidate_cycles(ldp.residual, cycles)
    validate_time = time.perf_counter() - validate_start

    dp_start = time.perf_counter()
    dp = solve_by_conflict_dp(cycles, ldp.remaining_budget)
    dp_wall = time.perf_counter() - dp_start

    full = None
    cand = None
    full_wall = 0.0
    cand_wall = 0.0
    if run_ilp:
        full_start = time.perf_counter()
        full = solve_residual_circulation_ilp(ldp.residual, ldp.remaining_budget, time_limit=gurobi_time_limit)
        full_wall = time.perf_counter() - full_start
        cand_start = time.perf_counter()
        cand = solve_candidate_edge_subgraph_ilp(ldp.residual, cycles, ldp.remaining_budget, time_limit=gurobi_time_limit)
        cand_wall = time.perf_counter() - cand_start

    result.update(
        {
            "cycles": cycles,
            "dp": dp,
            "full_ilp": full,
            "cand_ilp": cand,
            "enum_time": enum_time,
            "validate_time": validate_time,
            "dp_wall": dp_wall,
            "full_ilp_wall": full_wall,
            "cand_ilp_wall": cand_wall,
            "our_postprocess_time": enum_time + validate_time + dp_wall,
            "total_runtime": time.perf_counter() - total_start,
        }
    )
    return result


def _cycle_cost_summary(cycles) -> dict[tuple[str, int], int]:
    out: dict[tuple[str, int], int] = {}
    for cycle in cycles:
        key = (cycle.kind, cycle.cost)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda item: (item[0][0], item[0][1])))


def _print_experiment_result(result: dict[str, Any], show_cycles: bool = False, cycle_limit: int = 20) -> None:
    instance: CounterexampleInstance = result["instance"]
    G: nx.MultiDiGraph = result["graph"]
    ldp = result["ldp"]
    print(f"LDP feasible: {ldp.feasible}, message={ldp.message}")
    print(
        f"k={instance.k}, delta={instance.delta}, "
        f"slots_per_pair={instance.info['slots_per_pair']}, "
        f"release_alts={instance.info['candidate_release_cycles_at_least'] // max(1, instance.info['total_slots'])}, "
        f"use_alts={instance.info['candidate_use_cycles_at_least'] // max(1, instance.info['total_slots'])}"
    )
    print(f"graph_nodes={G.number_of_nodes()}, graph_edges={G.number_of_edges()}")
    print(
        f"theoretical_greedy={instance.info['greedy_pattern_weight']}, "
        f"theoretical_optimal={instance.info['optimal_pattern_weight']}, "
        f"expected_improvement={instance.info['expected_improvement']}"
    )
    print(f"base_weight={ldp.base_weight}, used_cost={ldp.used_cost}, remaining_budget={ldp.remaining_budget}")
    print(f"paths(edge ids)={ldp.paths}")
    print(f"LDP runtime={result['ldp_time']:.6f}s")
    if not ldp.feasible:
        print(f"total runtime={result['total_runtime']:.6f}s")
        return

    cycles = result["cycles"]
    print(f"num_candidates={len(cycles)}")
    print(f"candidate cost summary={_cycle_cost_summary(cycles)}")
    print(f"candidate enumeration runtime={result['enum_time']:.6f}s")
    print(f"candidate validation runtime={result['validate_time']:.6f}s")

    if show_cycles:
        edge_map = edge_by_id(ldp.residual)
        for i, cycle in enumerate(cycles[:cycle_limit]):
            print(f"cycle[{i}] kind={cycle.kind}, weight={cycle.weight}, cost={cycle.cost}")
            print(f"  {describe_cycle(edge_map, cycle.edge_ids)}")
        if len(cycles) > cycle_limit:
            print(f"... skipped {len(cycles) - cycle_limit} cycles")

    dp = result["dp"]
    print("-----------------测试总运行时间--------------------")
    print(f"total runtime={result['total_runtime']:.6f}s")

    print("-----------------ldp运行时间--------------------")
    print(f"LDP runtime={result['ldp_time']:.6f}s")
    print("-----------------our algorithm--------------------")
    print(f"candidate enumeration runtime={result['enum_time']:.6f}s")
    print(f"candidate validation runtime={result['validate_time']:.6f}s")
    print(f"DP runtime={dp.runtime_sec:.6f}s, wall={result['dp_wall']:.6f}s")
    print(f"our postprocess total runtime={result['our_postprocess_time']:.6f}s")
    print(f"LDP + our postprocess runtime={result['ldp_time'] + result['our_postprocess_time']:.6f}s")

    print("-----------------ILP algorithm--------------------")
    full = result["full_ilp"]
    cand = result["cand_ilp"]
    if full is None or cand is None:
        print("ILP skipped")
    else:
        print(f"full ILP runtime={full.runtime_sec:.6f}s, wall={result['full_ilp_wall']:.6f}s")
        print(f"cand ILP runtime={cand.runtime_sec:.6f}s, wall={result['cand_ilp_wall']:.6f}s")
        print(f"LDP + full ILP runtime={result['ldp_time'] + result['full_ilp_wall']:.6f}s")
        print(f"LDP + cand ILP runtime={result['ldp_time'] + result['cand_ilp_wall']:.6f}s")

    print("-----------------输出值对比--------------------")
    print(f"DP objective={dp.objective}, cost={dp.total_cost}, improved={dp.improved}, selected={len(dp.selected_cycles)}")
    if full is not None and cand is not None:
        print(f"full ILP objective={full.objective}, cost={full.total_cost}, improved={full.improved}, status={full.status}")
        print(f"cand ILP objective={cand.objective}, cost={cand.total_cost}, improved={cand.improved}, status={cand.status}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the contradiction counterexample through the LDP framework.")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--slots-per-pair", type=int, default=3)
    parser.add_argument("--release-alts", type=int, default=2)
    parser.add_argument("--use-alts", type=int, default=2)
    parser.add_argument("--alpha", type=int, default=2)
    parser.add_argument("--M", type=int, default=40)
    parser.add_argument("--max-cycles-per-anchor", type=int, default=None)
    parser.add_argument("--gurobi-time-limit", type=float, default=5.0)
    parser.add_argument("--skip-ilp", action="store_true")
    parser.add_argument("--show-cycles", action="store_true")
    parser.add_argument("--cycle-limit", type=int, default=20)
    args = parser.parse_args()

    result = run_counterexample_experiment(
        k=args.k,
        slots_per_pair=args.slots_per_pair,
        release_alts=args.release_alts,
        use_alts=args.use_alts,
        alpha=args.alpha,
        M=args.M,
        max_cycles_per_anchor=args.max_cycles_per_anchor,
        gurobi_time_limit=args.gurobi_time_limit,
        run_ilp=not args.skip_ilp,
    )
    _print_experiment_result(result, show_cycles=args.show_cycles, cycle_limit=args.cycle_limit)


if __name__ == "__main__":
    main()
