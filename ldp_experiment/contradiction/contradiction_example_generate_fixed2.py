"""A more robust planted LDP budget-swap counterexample generator.

Interface kept compatible with the previous scripts:
    generate_k_path_swap_counterexample(...) -> (edges, k, delta, info)

Key change from fixed_single_core_with_fillers_first:
    The release region x_i and the use region y_i are separated in the same
    way as the original hand-made 3-path counterexample:

        P1: ... b_i -> c_i -> d_i -> e_i ...
        P2: ... z_i -> h_i -> c_i -> j_i ...
        P3: ... z_i -> q_i -> r_i ... or q_i -> e_i -> r_i ...

    After P1_bad and P2 are chosen, P3_exp is feasible using only z_i.
    P3_cheap would also use e_i, but the budget is exhausted. Releasing c_i
    is a separate residual cycle around b_i-c_i-d_i and is not embedded as a
    forward detour of the P3 augmenting path.

This is intended to keep the LDP output near the planted greedy pattern, while
post-processing can combine release cycles and use cycles.
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
    edges: list[WeightedEdge]
    k: int
    delta: int
    info: dict[str, Any]

    def as_legacy_tuple(self) -> tuple[list[WeightedEdge], int, int, dict[str, Any]]:
        return self.edges, self.k, self.delta, self.info


def generate_k_path_swap_counterexample(
    k: int,
    slots_per_pair: int = 3,
    release_alts: int = 2,
    use_alts: int = 2,
    alpha: int = 2,
    M: int = 40,
) -> tuple[list[WeightedEdge], int, int, dict[str, Any]]:
    """Generate a robust multi-slot version of the original 3-path counterexample.

    There are k-3 isolated filler paths plus one 3-path core.

    For each slot i:
      - P1_bad and P2 share c_i, a bad common node.
      - P2 and P3 share z_i, an unavoidable common node.
      - P1_good bypasses c_i using b_i -> bypass_i_a -> d_i.
      - P3_cheap uses e_i through q_i -> ... -> e_i -> ... -> r_i.

    Budget delta = 2m. The LDP bad pattern uses {c_i, z_i}; the optimal pattern
    uses {e_i, z_i}. Both satisfy the restricted node-sharing constraint.
    """

    if k < 3:
        raise ValueError("k must be at least 3.")
    if slots_per_pair < 1:
        raise ValueError("slots_per_pair must be at least 1.")
    if release_alts < 1:
        raise ValueError("release_alts must be at least 1.")
    if use_alts < 1:
        raise ValueError("use_alts must be at least 1.")
    if M <= 4 + alpha:
        raise ValueError("M should be larger than 4 + alpha to create improvement.")

    edges: list[WeightedEdge] = []

    def add(u: str, v: str, w: int) -> None:
        edges.append((u, v, w))

    m = slots_per_pair
    delta = 2 * m
    filler_count = k - 3

    # ------------------------------------------------------------
    # 0. Cheap isolated filler paths, selected first.
    # ------------------------------------------------------------
    filler_paths: list[list[str]] = []
    for h in range(1, filler_count + 1):
        u = f"F_{h}_0"
        v = f"F_{h}_1"
        add("s", u, 1)
        add(u, v, 1)
        add(v, "t", 1)
        filler_paths.append(["s", u, v, "t"])
    filler_weight_total = 3 * filler_count

    # ------------------------------------------------------------
    # 1. P1: bad path through c_i; good path bypasses c_i but still uses e_i.
    # Pattern per slot:
    #   prev -> b_i -> c_i -> d_i -> e_i
    # bypass:
    #   b_i -> bypass_i_a -> d_i
    # ------------------------------------------------------------
    P1_bad = ["s"]
    P1_good = ["s"]
    prev = "s"
    for i in range(1, m + 1):
        b = f"b_{i}"
        c = f"c_{i}"
        d = f"d_{i}"
        e = f"e_{i}"

        add(prev, b, 1)
        add(b, c, 1)
        add(c, d, 1)
        add(d, e, 1)

        for a in range(1, release_alts + 1):
            x = f"bypass_{i}_{a}"
            # b -> c -> d has weight 2. Best bypass has weight 2 + alpha.
            add(b, x, 1)
            add(x, d, 1 + alpha + (a - 1))

        P1_bad.extend([b, c, d, e])
        P1_good.extend([b, f"bypass_{i}_1", d, e])
        prev = e

    add(prev, "t", 1)
    P1_bad.append("t")
    P1_good.append("t")

    # per slot: prev->b, b->c, c->d, d->e = 4; plus final e_m->t = 1
    P1_bad_weight = 4 * m + 1
    P1_good_weight = P1_bad_weight + alpha * m

    # ------------------------------------------------------------
    # 2. P2: shares c_i with P1_bad and z_i with P3.
    # Pattern per slot:
    #   prev -> z_i -> h_i -> c_i -> j_i
    # ------------------------------------------------------------
    P2 = ["s", "P2_0"]
    p2_gate = P1_bad_weight + 10
    add("s", "P2_0", p2_gate)

    prev = "P2_0"
    for i in range(1, m + 1):
        z = f"z_{i}"
        h = f"h_{i}"
        c = f"c_{i}"
        j = f"j_{i}"

        add(prev, z, 1)
        add(z, h, 1)
        add(h, c, 1)
        add(c, j, 1)

        P2.extend([z, h, c, j])
        prev = j

    add(prev, "t", 1)
    P2.append("t")
    P2_weight = p2_gate + 4 * m + 1

    # ------------------------------------------------------------
    # 3. P3: shares z_i with P2. Expensive edge q_i->r_i can be replaced
    # by cheap alternatives through e_i, but that would additionally consume
    # the good common nodes e_i.
    # Pattern per slot:
    #   prev -> z_i -> q_i -> r_i
    # cheap:
    #   q_i -> use_in_i_b -> e_i -> use_out_i_b -> r_i
    # ------------------------------------------------------------
    P3_exp = ["s", "P3_0"]
    P3_cheap = ["s", "P3_0"]
    p3_gate = P2_weight + 10
    add("s", "P3_0", p3_gate)

    prev = "P3_0"
    for i in range(1, m + 1):
        z = f"z_{i}"
        q = f"q_{i}"
        e = f"e_{i}"
        r = f"r_{i}"

        add(prev, z, 1)
        add(z, q, 1)
        add(q, r, M)

        for b_alt in range(1, use_alts + 1):
            inn = f"use_in_{i}_{b_alt}"
            out = f"use_out_{i}_{b_alt}"
            # Total cheap replacement q -> ... -> r has weight 4 + (b_alt - 1).
            add(q, inn, 1)
            add(inn, e, 1 + (b_alt - 1))
            add(e, out, 1)
            add(out, r, 1)

        P3_exp.extend([z, q, r])
        P3_cheap.extend([z, q, f"use_in_{i}_1", e, f"use_out_{i}_1", r])
        prev = r

    add(prev, "t", 1)
    P3_exp.append("t")
    P3_cheap.append("t")

    cheap_alt_base = 4
    P3_exp_weight = p3_gate + m * (M + 2) + 1
    P3_cheap_weight = p3_gate + m * (cheap_alt_base + 2) + 1

    greedy_weight = filler_weight_total + P1_bad_weight + P2_weight + P3_exp_weight
    optimal_pattern_weight = filler_weight_total + P1_good_weight + P2_weight + P3_cheap_weight
    expected_improvement = greedy_weight - optimal_pattern_weight

    info: dict[str, Any] = {
        "construction": "fixed2_separated_release_and_use_core",
        "g_pairs": 1,
        "slots_per_pair": m,
        "total_slots": m,
        "delta": delta,
        "filler_count": filler_count,
        "filler_paths": filler_paths,
        "P0_bad": P1_bad,
        "P0_good": P1_good,
        "P1_bad": P1_bad,
        "P1_good": P1_good,
        "P2": P2,
        "P3_expensive": P3_exp,
        "P3_cheap": P3_cheap,
        "blocker_paths": [P2],
        "saver_expensive_paths": [P3_exp],
        "saver_cheap_paths": [P3_cheap],
        "P0_bad_weight": P1_bad_weight,
        "P0_good_weight": P1_good_weight,
        "P1_bad_weight": P1_bad_weight,
        "P1_good_weight": P1_good_weight,
        "P2_weight": P2_weight,
        "P3_expensive_weight": P3_exp_weight,
        "P3_cheap_weight": P3_cheap_weight,
        "blocker_weight_each": P2_weight,
        "saver_expensive_weight_each": P3_exp_weight,
        "saver_cheap_weight_each": P3_cheap_weight,
        "greedy_pattern_weight": greedy_weight,
        "optimal_pattern_weight": optimal_pattern_weight,
        "expected_improvement": expected_improvement,
        "per_slot_improvement": M - cheap_alt_base - alpha,
        "candidate_release_cycles_at_least": m * release_alts,
        "candidate_use_cycles_at_least": m * use_alts,
        "candidate_cycles_at_least": m * (release_alts + use_alts),
        "bad_common_nodes": [f"c_{i}" for i in range(1, m + 1)],
        "good_common_nodes": [f"e_{i}" for i in range(1, m + 1)],
        "z_common_nodes": [f"z_{i}" for i in range(1, m + 1)],
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
    G = nx.MultiDiGraph()
    for u, v, weight in edges:
        add_edge_with_attrs(G, u, v, weight=weight, cost=0, desc="fixed2_contradiction")
    return G


def describe_cycle(edge_map, edge_ids: tuple[int, ...]) -> str:
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
    info = instance.info

    print(f"LDP feasible: {ldp.feasible}, message={ldp.message}")
    print(
        f"k={instance.k}, delta={instance.delta}, "
        f"slots_per_pair={info['slots_per_pair']}, "
        f"release_alts={info['candidate_release_cycles_at_least'] // max(1, info['total_slots'])}, "
        f"use_alts={info['candidate_use_cycles_at_least'] // max(1, info['total_slots'])}"
    )
    print(f"construction={info.get('construction')}")
    print(f"graph_nodes={G.number_of_nodes()}, graph_edges={G.number_of_edges()}")
    print(
        f"theoretical_greedy={info['greedy_pattern_weight']}, "
        f"theoretical_optimal={info['optimal_pattern_weight']}, "
        f"expected_improvement={info['expected_improvement']}"
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
    parser = argparse.ArgumentParser(description="Run the fixed2 contradiction counterexample through the LDP framework.")
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
