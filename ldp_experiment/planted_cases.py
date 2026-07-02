"""Parameterized planted residual-improvement instances."""

from __future__ import annotations

import argparse
import random
import time
from typing import Optional

import networkx as nx

from ldp_experiment.candidate_cycles import enumerate_candidate_cycles, validate_candidate_cycles
from ldp_experiment.conflict_dp import solve_by_conflict_dp
from ldp_experiment.graph_utils import add_edge_with_attrs, edge_by_id
from ldp_experiment.ldp_algorithm import run_ldp
from ldp_experiment.residual_ilp import solve_candidate_edge_subgraph_ilp, solve_residual_circulation_ilp


def generate_planted_improvement_graph(
    k: int,
    *,
    blocks: int = 1,
    release_variants: int = 1,
    improve_variants: int = 1,
    release_span: int = 1,
    improve_span: int = 1,
    diverse: bool = False,
    extra_path_length: int = 4,
    high_edge_weight: int = 20,
    detour_left_weight: int = 1,
    detour_right_weight: int = 2,
    release_shortcut_weight: int = 4,
    segment_weight: int = 1,
    noise_nodes: int = 0,
    noise_edges: int = 0,
    noise_weight_low: int = 30,
    noise_weight_high: int = 100,
    allow_parallel: bool = False,
    seed: Optional[int] = None,
) -> nx.MultiDiGraph:
    """Generate a graph with planted release-budget and negative-weight cycles.

    The construction generalizes ``manual_cases2`` with a stable three-path
    improvement core. ``blocks`` multiplies the number of release and improve
    variants inside that one core, which increases candidate-cycle count, node
    count, and edge count without letting LDP consume independent blocks before
    the postprocess phase. Extra disjoint paths increase k.
    """
    if blocks < 1:
        raise ValueError("blocks must be at least 1")
    if release_variants < 1 or improve_variants < 1:
        raise ValueError("release_variants and improve_variants must be at least 1")
    if release_span < 1 or improve_span < 1:
        raise ValueError("release_span and improve_span must be at least 1")
    if extra_path_length < 1:
        raise ValueError("extra_path_length must be at least 1")
    required_paths = 3
    if k < required_paths:
        raise ValueError("k must be at least 3")
    if noise_edges < 0 or noise_nodes < 0:
        raise ValueError("noise sizes must be non-negative")

    rng = random.Random(seed)
    G = nx.MultiDiGraph()
    used_pairs: set[tuple[str, str]] = set()

    def add(u: str, v: str, weight: int) -> None:
        if not allow_parallel and (u, v) in used_pairs:
            raise ValueError(f"duplicate planted edge {u}->{v}")
        used_pairs.add((u, v))
        add_edge_with_attrs(G, u, v, weight=weight, cost=0, desc="planted")

    total_release_variants = blocks * release_variants
    total_improve_variants = blocks * improve_variants

    if diverse and False:
        release_sites = [
            ("b0", "c1", "d0"),
            ("mid_1_0", "c2", "mid_1_1"),
            ("mid_1_1", "c1", "mid_1_2"),
        ]
        improve_targets = ["e", "d0", "mid_1_3", "mid_1_4", "a0"]

        # Path 0 supplies several possible detour targets for the long path.
        add("s", "a0", segment_weight)
        add("a0", "b0", segment_weight)
        add("b0", "c1", segment_weight)
        add("c1", "d0", segment_weight)
        add("d0", "e", segment_weight)
        add("e", "path0_tail_0", segment_weight)
        add("path0_tail_0", "t", segment_weight)

        # Middle path shares c1 with path 0 and c2 with the long path. Release
        # shortcuts are placed on different path segments, so the resulting
        # negative-cost cycles use different reverse anchors and split nodes.
        add("s", "mid_1_0", segment_weight)
        add("mid_1_0", "c2", segment_weight)
        add("c2", "mid_1_1", segment_weight)
        add("mid_1_1", "c1", segment_weight)
        add("c1", "mid_1_2", segment_weight)
        add("mid_1_2", "mid_1_3", segment_weight)
        add("mid_1_3", "mid_1_4", segment_weight)
        add("mid_1_4", "t", segment_weight)

        for i in range(total_release_variants):
            left, _middle, right = release_sites[i % len(release_sites)]
            if i < len(release_sites):
                add(left, right, release_shortcut_weight + i)
            else:
                rel = f"diverse_release_alt_{i}"
                add(left, rel, segment_weight)
                add(rel, right, release_shortcut_weight + i - segment_weight)

        # Long path contains several distinct high edges. Each cheaper detour
        # targets a different already-used node, so it would add one extra
        # common node during LDP and is blocked when delta is tight.
        add("s", "long_0", segment_weight)
        add("long_0", "c2", segment_weight)
        prev = "c2"
        for i in range(total_improve_variants):
            q = f"diverse_q_{i}"
            r = f"diverse_r_{i}"
            target = improve_targets[i % len(improve_targets)]
            add(prev, q, segment_weight)
            add(q, r, high_edge_weight + i)
            add(r, f"long_after_diverse_{i}", segment_weight)
            prev = f"long_after_diverse_{i}"
            if i < len(improve_targets):
                add(q, target, detour_left_weight + (i % 2))
            else:
                imp = f"diverse_improve_alt_{i}"
                add(q, imp, segment_weight)
                add(imp, target, detour_left_weight + i)
            add(target, r, detour_right_weight + (i % 3))
        add(prev, "long_tail_0", segment_weight)
        add("long_tail_0", "long_tail_1", segment_weight)
        add("long_tail_1", "t", segment_weight)

        for i in range(required_paths, k):
            prev = "s"
            for j in range(extra_path_length - 1):
                cur = f"extra_{i}_{j}"
                add(prev, cur, segment_weight)
                prev = cur
            add(prev, "t", segment_weight)

        noise_vertices = [f"noise_{i}" for i in range(noise_nodes)]
        all_nodes = list(G.nodes) + noise_vertices
        G.add_nodes_from(noise_vertices)
        attempts = 0
        while G.number_of_edges() < len(used_pairs) + noise_edges and attempts < max(100, 20 * max(1, noise_edges)):
            attempts += 1
            u = rng.choice(all_nodes)
            v = rng.choice(all_nodes)
            if u == v:
                continue
            if not allow_parallel and (u, v) in used_pairs:
                continue
            used_pairs.add((u, v))
            add_edge_with_attrs(
                G,
                u,
                v,
                weight=rng.randint(noise_weight_low, noise_weight_high),
                cost=0,
                desc="noise",
            )
        return G

    release_nodes = ["c1"] + [f"release_c{i}" for i in range(2, release_span + 1)]
    improve_nodes = ["e"] + [f"improve_e{i}" for i in range(2, improve_span + 1)]
    structural_release_shortcut_weight = max(release_shortcut_weight, 2 * release_span + 2)

    # Path 0 has a release segment b0 -> release_nodes -> d0 and later the
    # improve detour target nodes. A shortcut b0 -> d0 crosses every release
    # node in the residual cycle, so its negative cost is structural.
    add("s", "a0", segment_weight)
    add("a0", "b0", segment_weight)
    prev = "b0"
    for node in release_nodes:
        add(prev, node, segment_weight)
        prev = node
    add(prev, "d0", segment_weight)
    prev = "d0"
    for node in improve_nodes:
        add(prev, node, segment_weight)
        prev = node
    add(prev, "t", segment_weight)

    # Release variants generate many nonnegative-weight negative-cost cycles.
    # They share the same reversed b0-c1 and c1-d0 edges, so they intentionally
    # form one conflict block.
    for variant in range(total_release_variants):
        if not diverse and variant == 0:
            add("b0", "d0", structural_release_shortcut_weight)
        elif not diverse:
            rel = f"release_alt_{variant}"
            add("b0", rel, segment_weight)
            total_weight = structural_release_shortcut_weight + variant
            add(rel, "d0", total_weight - segment_weight)
        else:
            release_sites = [("b0", "d0")]
            release_sites.extend(
                (f"mid_before_{index}", f"mid_after_{index}")
                for index in range(release_span)
            )
            left, right = release_sites[variant % len(release_sites)]
            chain_len = 1 + (variant % 3)
            total_weight = structural_release_shortcut_weight + variant
            if chain_len == 1 and variant < len(release_sites):
                add(left, right, total_weight)
            elif chain_len == 1:
                rel = f"release_alt_{variant}_0"
                add(left, rel, segment_weight)
                add(rel, right, total_weight - segment_weight)
            else:
                prev = left
                for step in range(chain_len):
                    cur = right if step == chain_len - 1 else f"release_alt_{variant}_{step}"
                    weight = total_weight if step == chain_len - 1 else segment_weight
                    add(prev, cur, weight)
                    prev = cur

    # Middle core path. It shares every release node with path 0 and c2 with the
    # long path. The base LDP solution therefore uses release_span + 1 budget.
    add("s", "mid_1_0", segment_weight)
    add("mid_1_0", "c2", segment_weight)
    prev = "c2"
    for index, node in enumerate(reversed(release_nodes)):
        before = f"mid_before_{index}"
        after = f"mid_after_{index}"
        add(prev, before, segment_weight)
        add(before, node, segment_weight)
        add(node, after, segment_weight)
        prev = after
    add(prev, "mid_1_3", segment_weight)
    add("mid_1_3", "mid_1_4", segment_weight)
    add("mid_1_4", "t", segment_weight)

    # Long path uses q->r. Cheaper q->...->e->r variants are blocked during LDP
    # because e is already used by path 0 and the core budget is full.
    add("s", "long_0", segment_weight)
    add("long_0", "c2", segment_weight)
    add("c2", "long_1", segment_weight)
    add("long_1", "long_2", segment_weight)
    add("long_2", "long_3", segment_weight)
    add("long_3", "q", segment_weight)
    add("q", "r", high_edge_weight)
    add("r", "long_4", segment_weight)
    add("long_4", "long_5", segment_weight)
    add("long_5", "t", segment_weight)

    for index in range(len(improve_nodes) - 1):
        aux = f"improve_detour_link_{index}"
        add(improve_nodes[index], aux, segment_weight)
        add(aux, improve_nodes[index + 1], segment_weight)
    add(improve_nodes[-1], "r", detour_right_weight)
    for variant in range(total_improve_variants):
        if not diverse and variant == 0:
            add("q", improve_nodes[0], detour_left_weight)
        elif not diverse:
            imp = f"improve_alt_{variant}"
            add("q", imp, segment_weight)
            add(imp, improve_nodes[0], detour_left_weight + variant)
        else:
            targets = [improve_nodes[0]]
            target = targets[variant % len(targets)]
            chain_len = 1 + (variant % 3)
            if chain_len == 1 and variant > 0:
                chain_len = 2
            prev = "q"
            for step in range(chain_len):
                cur = target if step == chain_len - 1 else f"improve_alt_{variant}_{step}"
                weight = detour_left_weight + variant if step == chain_len - 1 else segment_weight
                add(prev, cur, weight)
                prev = cur

    # Extra paths are intentionally disjoint from the improvement blocks. They
    # let experiments scale k while preserving a predictable planted optimum.
    for i in range(required_paths, k):
        prev = "s"
        for j in range(extra_path_length - 1):
            cur = f"extra_{i}_{j}"
            add(prev, cur, segment_weight)
            prev = cur
        add(prev, "t", segment_weight)

    # Optional high-weight random noise.
    noise_vertices = [f"noise_{i}" for i in range(noise_nodes)]
    all_nodes = list(G.nodes) + noise_vertices
    G.add_nodes_from(noise_vertices)
    attempts = 0
    while G.number_of_edges() < len(used_pairs) + noise_edges and attempts < max(100, 20 * max(1, noise_edges)):
        attempts += 1
        u = rng.choice(all_nodes)
        v = rng.choice(all_nodes)
        if u == v:
            continue
        if not allow_parallel and (u, v) in used_pairs:
            continue
        used_pairs.add((u, v))
        add_edge_with_attrs(
            G,
            u,
            v,
            weight=rng.randint(noise_weight_low, noise_weight_high),
            cost=0,
            desc="noise",
        )
    return G


def describe_cycle(edge_map, edge_ids: tuple[int, ...]) -> str:
    parts = []
    for eid in edge_ids:
        edge = edge_map[eid]
        marker = "R" if edge.is_reverse else "F"
        parts.append(f"{eid}:{edge.u}->{edge.v}:w={edge.weight}:c={edge.cost}:{marker}")
    return " | ".join(parts)


def apply_varied_split_costs(
    R: nx.MultiDiGraph,
    pattern: tuple[int, ...] = (4, 3, 2, 5),
) -> dict[str, int]:
    """Assign heterogeneous costs to nonzero split edges in a residual graph.

    ``run_ldp`` currently creates every split edge with unit cost. This helper
    is an experiment-side residual perturbation: it keeps edge directions and
    weights unchanged, but changes each nonzero split edge cost according to
    its base node. Reverse split edges keep the negative sign, so candidate
    cycles can have costs such as -3, -4, +4, etc.
    """
    if not pattern or any(value <= 0 for value in pattern):
        raise ValueError("split cost pattern must contain positive integers")

    def split_base(u, v) -> Optional[str]:
        bases = []
        for node in (u, v):
            if isinstance(node, tuple) and len(node) == 2 and node[1] in {"in", "out"}:
                bases.append(str(node[0]))
        if bases and all(base == bases[0] for base in bases):
            return bases[0]
        return None

    bases: list[str] = []
    for u, v, _key, data in R.edges(keys=True, data=True):
        if not data.get("is_split_edge") or int(data.get("cost", 0)) == 0:
            continue
        base = split_base(u, v)
        if base is not None and base not in bases:
            bases.append(base)

    preferred = {
        "c1": 4,
        "c2": 3,
        "e": 4,
        "d0": 2,
        "mid_1_3": 5,
        "mid_1_4": 3,
    }
    assigned: dict[str, int] = {}
    for index, base in enumerate(sorted(bases)):
        assigned[base] = preferred.get(base, pattern[index % len(pattern)])

    for u, v, _key, data in R.edges(keys=True, data=True):
        old_cost = int(data.get("cost", 0))
        if not data.get("is_split_edge") or old_cost == 0:
            continue
        base = split_base(u, v)
        if base is None:
            continue
        sign = 1 if old_cost > 0 else -1
        data["cost"] = sign * assigned[base]
    return assigned


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a planted residual-improvement instance.")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--delta", type=int, default=None)
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--release-variants", type=int, default=1)
    parser.add_argument("--improve-variants", type=int, default=1)
    parser.add_argument("--release-span", type=int, default=1)
    parser.add_argument("--improve-span", type=int, default=1)
    parser.add_argument("--diverse", action="store_true")
    parser.add_argument("--varied-costs", action="store_true")
    parser.add_argument("--split-cost-pattern", type=str, default="4,3,2,5")
    parser.add_argument("--extra-path-length", type=int, default=4)
    parser.add_argument("--noise-nodes", type=int, default=0)
    parser.add_argument("--noise-edges", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gurobi-time-limit", type=float, default=5.0)
    parser.add_argument("--max-cycles-per-anchor", type=int, default=None)
    args = parser.parse_args()

    delta = args.release_span + 1 if args.delta is None else args.delta
    total_start = time.perf_counter()
    G = generate_planted_improvement_graph(
        args.k,
        blocks=args.blocks,
        release_variants=args.release_variants,
        improve_variants=args.improve_variants,
        release_span=args.release_span,
        improve_span=args.improve_span,
        diverse=args.diverse,
        extra_path_length=args.extra_path_length,
        noise_nodes=args.noise_nodes,
        noise_edges=args.noise_edges,
        seed=args.seed,
    )

    ldp_start = time.perf_counter()
    ldp = run_ldp(G, "s", "t", args.k, delta)
    ldp_time = time.perf_counter() - ldp_start
    print(f"LDP feasible: {ldp.feasible}, message={ldp.message}")
    print(
        f"k={args.k}, delta={delta}, blocks={args.blocks}, "
        f"release_variants={args.release_variants}, improve_variants={args.improve_variants}, "
        f"release_span={args.release_span}, improve_span={args.improve_span}, "
        f"diverse={args.diverse}"
    )
    print(f"graph_nodes={G.number_of_nodes()}, graph_edges={G.number_of_edges()}")
    print(f"base_weight={ldp.base_weight}, used_cost={ldp.used_cost}, remaining_budget={ldp.remaining_budget}")
    print(f"paths(edge ids)={ldp.paths}")
    print(f"LDP runtime={ldp_time:.6f}s")
    if not ldp.feasible:
        print(f"total runtime={time.perf_counter() - total_start:.6f}s")
        return

    if args.varied_costs:
        pattern = tuple(int(part.strip()) for part in args.split_cost_pattern.split(",") if part.strip())
        split_costs = apply_varied_split_costs(ldp.residual, pattern)
        print(f"varied_split_costs={split_costs}")

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
    dp_wall = time.perf_counter() - dp_start
    postprocess_time = enum_time + dp_wall

    full_start = time.perf_counter()
    full = solve_residual_circulation_ilp(ldp.residual, ldp.remaining_budget, time_limit=args.gurobi_time_limit)
    full_wall = time.perf_counter() - full_start
    cand_start = time.perf_counter()
    cand = solve_candidate_edge_subgraph_ilp(ldp.residual, cycles, ldp.remaining_budget, time_limit=args.gurobi_time_limit)
    cand_wall = time.perf_counter() - cand_start

    print("-----------------测试总运行时间--------------------")
    print(f"total runtime={time.perf_counter() - total_start:.6f}s")

    print("-----------------ldp运行时间--------------------")
    print(f"LDP runtime={ldp_time:.6f}s")
    print("-----------------our algorithm--------------------")
    print(f"candidate enumeration runtime={enum_only_time:.6f}s")
    print(f"candidate validation runtime={validate_time:.6f}s")
    print(f"DP runtime={dp.runtime_sec:.6f}s, wall={dp_wall:.6f}s")
    print(f"our postprocess total runtime={postprocess_time:.6f}s")
    print(f"LDP + our postprocess runtime={ldp_time + postprocess_time:.6f}s")
    print("-----------------ILP algorithm--------------------")
    print(f"full ILP runtime={full.runtime_sec:.6f}s, wall={full_wall:.6f}s")
    print(f"cand ILP runtime={cand.runtime_sec:.6f}s, wall={cand_wall:.6f}s")
    print(f"LDP + full ILP runtime={ldp_time + full_wall:.6f}s")
    print(f"LDP + cand ILP runtime={ldp_time + cand_wall:.6f}s")

    print("-----------------输出值对比--------------------")
    print(f"DP objective={dp.objective}, cost={dp.total_cost}, improved={dp.improved}, selected={len(dp.selected_cycles)}")
    print(f"full ILP objective={full.objective}, cost={full.total_cost}, improved={full.improved}, status={full.status}")
    print(f"cand ILP objective={cand.objective}, cost={cand.total_cost}, improved={cand.improved}, status={cand.status}")


if __name__ == "__main__":
    main()
