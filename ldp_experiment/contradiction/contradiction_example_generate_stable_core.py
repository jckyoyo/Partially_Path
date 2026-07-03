"""Stable planted LDP budget-swap counterexamples.

This generator deliberately keeps the *original hand-made 3-path failure core*
and only adds parallel local alternatives to increase the number of candidate
cycles.  Extra paths for arbitrary k are isolated filler paths, so they do not
create new residual channels that allow LDP to repair the core early.

Public legacy interface is preserved:
    generate_k_path_swap_counterexample(...) -> (edges, k, delta, info)
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
        return self.edges, self.k, self.delta, self.info


def generate_k_path_swap_counterexample(
    k: int,
    slots_per_pair: int = 6,
    release_alts: int = 2,
    use_alts: int = 2,
    alpha: int = 2,
    M: int = 40,
) -> tuple[list[WeightedEdge], int, int, dict[str, Any]]:
    """Generate a stable LDP counterexample for arbitrary k.

    The core is the user's original 3-path counterexample shape:

        P1_bad:  s-a-b-c-d-e-t
        P2:      s-f-g-h-c-j-k-l-t
        P3_exp:  s-m-g-n-o-p-q-r-v-u-t

    LDP with delta=2 first consumes common nodes {c,g} and is forced to use
    q->r.  After q->r is selected, a negative positive-cost use cycle through e
    appears; it can be combined with a nonnegative negative-cost release cycle
    around c.

    To increase candidate cycles without changing this timing property, we add
    many *parallel* release alternatives b -> rel -> d and many *parallel* use
    alternatives q -> use_in -> e -> use_out -> r.  These alternatives all
    share the same underlying bad/good common nodes, so only a small number are
    selected, but the candidate cycle set can be large.

    For k>3, we add k-3 isolated filler paths.  They are chosen before the core
    because they are cheaper, and they do not connect to the core except at s,t.
    """
    if k < 3:
        raise ValueError("k must be at least 3")
    if slots_per_pair < 1:
        raise ValueError("slots_per_pair must be positive")
    if release_alts < 1 or use_alts < 1:
        raise ValueError("release_alts and use_alts must be positive")
    if M <= 4 + alpha:
        raise ValueError("M should be larger than 4 + alpha to make the swap improving")

    edges: list[WeightedEdge] = []

    def add(u: str, v: str, w: int) -> None:
        edges.append((u, v, int(w)))

    # ------------------------------------------------------------
    # 0. Isolated filler paths: chosen first, but irrelevant to core.
    # ------------------------------------------------------------
    filler_count = k - 3
    filler_paths: list[list[str]] = []
    filler_weight_each = 3
    for h in range(1, filler_count + 1):
        x = f"F_{h}_x"
        y = f"F_{h}_y"
        add("s", x, 1)
        add(x, y, 1)
        add(y, "t", 1)
        filler_paths.append(["s", x, y, "t"])

    # ------------------------------------------------------------
    # 1. Stable 3-path core.
    # ------------------------------------------------------------
    # P1_bad: s-a-b-c-d-e-t
    add("s", "a", 1)
    add("a", "b", 1)
    add("b", "c", 1)
    add("c", "d", 1)
    add("d", "e", 1)
    add("e", "t", 1)

    # Many parallel release alternatives around c: b -> rel -> d.
    # Best release alternative has total weight 2+alpha, so compared with
    # b->c->d of weight 2, releasing c costs alpha.
    release_nodes: list[str] = []
    for i in range(1, slots_per_pair + 1):
        for a_idx in range(1, release_alts + 1):
            rel = f"rel_{i}_{a_idx}"
            release_nodes.append(rel)
            add("b", rel, 1)
            add(rel, "d", 1 + alpha + (a_idx - 1))

    # P2: s-f-g-h-c-j-k-l-t.  It shares c with P1_bad.
    add("s", "f", 1)
    add("f", "g", 1)
    add("g", "h", 1)
    add("h", "c", 1)
    add("c", "j", 1)
    add("j", "k", 1)
    add("k", "l", 1)
    add("l", "t", 1)

    # P3_exp: s-m-g-n-o-p-q-r-v-u-t.  It shares g with P2 and is forced to
    # use q->r while budget is full.
    add("s", "m", 1)
    add("m", "g", 1)
    add("g", "n", 1)
    add("n", "o", 1)
    add("o", "p", 1)
    add("p", "q", 1)
    add("q", "r", M)
    add("r", "v", 1)
    add("v", "u", 1)
    add("u", "t", 1)

    # Many parallel use alternatives through e: q -> in -> e -> out -> r.
    # Best use alternative has total weight 4; compared with q->r of weight M,
    # using e saves M-4 but consumes one common-node budget.
    use_nodes: list[tuple[str, str]] = []
    for i in range(1, slots_per_pair + 1):
        for b_idx in range(1, use_alts + 1):
            inn = f"use_in_{i}_{b_idx}"
            out = f"use_out_{i}_{b_idx}"
            use_nodes.append((inn, out))
            add("q", inn, 1)
            add(inn, "e", 1 + (b_idx - 1))
            add("e", out, 1)
            add(out, "r", 1)

    # ------------------------------------------------------------
    # 2. Intended path descriptions and weights.
    # ------------------------------------------------------------
    p1_bad = ["s", "a", "b", "c", "d", "e", "t"]
    # Use the best release alternative for the theoretical optimal pattern.
    p1_good = ["s", "a", "b", "rel_1_1", "d", "e", "t"]
    p2 = ["s", "f", "g", "h", "c", "j", "k", "l", "t"]
    p3_exp = ["s", "m", "g", "n", "o", "p", "q", "r", "v", "u", "t"]
    p3_cheap = ["s", "m", "g", "n", "o", "p", "q", "use_in_1_1", "e", "use_out_1_1", "r", "v", "u", "t"]

    p1_bad_weight = 6
    p1_good_weight = 6 + alpha
    p2_weight = 8
    p3_exp_weight = M + 9
    p3_cheap_weight = 13  # prefix 6 + cheap detour 4 + suffix 3

    filler_total = filler_count * filler_weight_each
    greedy_weight = filler_total + p1_bad_weight + p2_weight + p3_exp_weight
    optimal_weight = filler_total + p1_good_weight + p2_weight + p3_cheap_weight
    expected_improvement = greedy_weight - optimal_weight

    # The restricted common-node budget is exactly enough for {c,g} in the
    # greedy pattern or {e,g} in the optimal pattern.
    delta = 2

    info: dict[str, Any] = {
        "construction": "stable_single_original_core_parallel_candidates",
        "slots_per_pair": slots_per_pair,
        "release_alts": release_alts,
        "use_alts": use_alts,
        "delta": delta,
        "filler_count": filler_count,
        "filler_paths": filler_paths,
        "filler_weight_each": filler_weight_each,
        "P1_bad": p1_bad,
        "P1_good": p1_good,
        "P2": p2,
        "P3_expensive": p3_exp,
        "P3_cheap": p3_cheap,
        "P1_bad_weight": p1_bad_weight,
        "P1_good_weight": p1_good_weight,
        "P2_weight": p2_weight,
        "P3_expensive_weight": p3_exp_weight,
        "P3_cheap_weight": p3_cheap_weight,
        "greedy_pattern_weight": greedy_weight,
        "optimal_pattern_weight": optimal_weight,
        "expected_improvement": expected_improvement,
        "per_best_swap_improvement": expected_improvement,
        "candidate_release_cycles_at_least": slots_per_pair * release_alts,
        "candidate_use_cycles_at_least": slots_per_pair * use_alts,
        "candidate_cycles_at_least": slots_per_pair * (release_alts + use_alts),
        "bad_common_nodes": ["c"],
        "good_common_nodes": ["e"],
        "fixed_common_nodes": ["g"],
        "release_nodes": release_nodes,
        "use_nodes": use_nodes,
    }
    return edges, k, delta, info


def generate_counterexample_instance(
    k: int,
    slots_per_pair: int = 6,
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
        add_edge_with_attrs(G, u, v, weight=weight, cost=0, desc="stable_core_counterexample")
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
    slots_per_pair: int = 6,
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
        f"k={instance.k}, delta={instance.delta}, slots_per_pair={info['slots_per_pair']}, "
        f"release_alts={info['release_alts']}, use_alts={info['use_alts']}"
    )
    print(f"construction={info['construction']}")
    print(f"graph_nodes={G.number_of_nodes()}, graph_edges={G.number_of_edges()}")
    print(
        f"theoretical_greedy={info['greedy_pattern_weight']}, "
        f"theoretical_optimal={info['optimal_pattern_weight']}, "
        f"expected_improvement={info['expected_improvement']}"
    )
    print(f"candidate_cycles_at_least={info['candidate_cycles_at_least']}")
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
    parser = argparse.ArgumentParser(description="Run a stable original-core contradiction counterexample.")
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--slots-per-pair", type=int, default=6)
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
