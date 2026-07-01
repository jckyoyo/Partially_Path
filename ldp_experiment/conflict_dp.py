"""Conflict graph decomposition, Pareto frontiers, and budget DP."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
import time
import warnings

import networkx as nx

from .candidate_cycles import Cycle
from .graph_utils import EPS


@dataclass
class DPResult:
    objective: float
    total_cost: int
    selected_cycles: list[Cycle]
    improved: bool
    num_cycles: int
    num_components: int
    max_component_size: int
    num_dp_states: int
    runtime_sec: float


@dataclass(frozen=True)
class BlockState:
    cost: int
    weight: float
    selected_indices: tuple[int, ...]


@dataclass
class _Node:
    cost: int
    weight: float
    prev: "_Node | None"
    action: BlockState | None


def _pareto_states(states: list[BlockState]) -> list[BlockState]:
    states = sorted(states, key=lambda s: (s.cost, s.weight, len(s.selected_indices)))
    frontier: list[BlockState] = []
    for state in states:
        dominated = False
        for kept in frontier:
            if kept.cost <= state.cost and kept.weight <= state.weight + EPS:
                dominated = True
                break
        if not dominated:
            frontier.append(state)
    return frontier


def _pareto_dp_nodes(dp: dict[int, _Node]) -> dict[int, _Node]:
    items = sorted(dp.items(), key=lambda item: (item[0], item[1].weight))
    out: dict[int, _Node] = {}
    best_weight = float("inf")
    for cost, node in items:
        if node.weight < best_weight - EPS:
            out[cost] = node
            best_weight = node.weight
    return out


def _solve_frontiers(frontiers: list[list[BlockState]], cycles: list[Cycle], B: int, meta: tuple[int, int, int], start: float) -> DPResult:
    releases = [max((-s.cost for s in f), default=0) for f in frontiers]
    suffix = [0] * (len(frontiers) + 1)
    for i in range(len(frontiers) - 1, -1, -1):
        suffix[i] = suffix[i + 1] + releases[i]
    root = _Node(0, 0.0, None, None)
    dp: dict[int, _Node] = {0: root}
    max_states = 1
    for j, frontier in enumerate(frontiers):
        new_dp: dict[int, _Node] = {}
        # Costs can temporarily exceed B: later negative-cost cycles may release
        # budget. The only safe pruning cap is B plus the remaining possible
        # release after this block.
        cap = B + suffix[j + 1]
        for old_cost, old_node in dp.items():
            for state in frontier:
                new_cost = old_cost + state.cost
                if new_cost > cap:
                    continue
                new_weight = old_node.weight + state.weight
                old_best = new_dp.get(new_cost)
                if old_best is None or new_weight < old_best.weight - EPS:
                    new_dp[new_cost] = _Node(new_cost, new_weight, old_node, state)
        dp = _pareto_dp_nodes(new_dp)
        max_states = max(max_states, len(dp))
    feasible = [(cost, node) for cost, node in dp.items() if cost <= B]
    if feasible:
        best_cost, best_node = min(feasible, key=lambda item: item[1].weight)
    else:
        best_cost, best_node = 0, root
    selected_indices: list[int] = []
    cur = best_node
    while cur is not None and cur.action is not None:
        selected_indices.extend(cur.action.selected_indices)
        cur = cur.prev
    selected = [cycles[i] for i in selected_indices]
    num_components, max_component_size, num_cycles = meta
    return DPResult(
        objective=best_node.weight,
        total_cost=best_cost,
        selected_cycles=selected,
        improved=best_node.weight < -EPS,
        num_cycles=num_cycles,
        num_components=num_components,
        max_component_size=max_component_size,
        num_dp_states=max_states,
        runtime_sec=time.perf_counter() - start,
    )


def _component_frontier(cycles: list[Cycle], indices: list[int], conflicts: dict[int, set[int]]) -> list[BlockState]:
    states: list[BlockState] = []
    chosen: list[int] = []

    def backtrack(pos: int, cost: int, weight: float) -> None:
        if pos == len(indices):
            states.append(BlockState(cost, weight, tuple(chosen)))
            return
        idx = indices[pos]
        backtrack(pos + 1, cost, weight)
        if all(idx not in conflicts[j] for j in chosen):
            chosen.append(idx)
            c = cycles[idx]
            backtrack(pos + 1, cost + c.cost, weight + c.weight)
            chosen.pop()

    backtrack(0, 0, 0.0)
    return _pareto_states(states)


def solve_by_conflict_dp(cycles: list[Cycle], B: int, max_exact_component_size: int = 25) -> DPResult:
    """Solve candidate-cycle selection under shared-edge conflicts."""
    start = time.perf_counter()
    C = nx.Graph()
    C.add_nodes_from(range(len(cycles)))
    conflicts: dict[int, set[int]] = {i: set() for i in range(len(cycles))}
    edge_to_cycles: dict[int, list[int]] = defaultdict(list)
    for i, cycle in enumerate(cycles):
        for eid in cycle.edge_ids:
            edge_to_cycles[eid].append(i)
    for ids in edge_to_cycles.values():
        for i, j in combinations(ids, 2):
            C.add_edge(i, j)
            conflicts[i].add(j)
            conflicts[j].add(i)
    components = [sorted(comp) for comp in nx.connected_components(C)]
    frontiers: list[list[BlockState]] = []
    for comp in components:
        if len(comp) > max_exact_component_size:
            warnings.warn(
                f"large conflict component size {len(comp)}; exact independent-set enumeration may be infeasible",
                RuntimeWarning,
            )
        frontiers.append(_component_frontier(cycles, comp, conflicts))
    meta = (len(components), max((len(c) for c in components), default=0), len(cycles))
    return _solve_frontiers(frontiers, cycles, B, meta, start)


def solve_by_knapsack_dp_no_conflict(cycles: list[Cycle], B: int) -> DPResult:
    """0-1 DP for pairwise edge-disjoint cycles, allowing negative item costs."""
    start = time.perf_counter()
    frontiers = [[BlockState(0, 0.0, ()), BlockState(c.cost, c.weight, (i,))] for i, c in enumerate(cycles)]
    frontiers = [_pareto_states(f) for f in frontiers]
    meta = (len(cycles), 1 if cycles else 0, len(cycles))
    return _solve_frontiers(frontiers, cycles, B, meta, start)
