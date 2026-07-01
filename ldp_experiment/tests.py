"""Small executable tests for the LDP experiment package."""

from __future__ import annotations

import os
import sys

import networkx as nx

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ldp_experiment.candidate_cycles import Cycle, enumerate_candidate_cycles, validate_candidate_cycles
from ldp_experiment.conflict_dp import solve_by_conflict_dp, solve_by_knapsack_dp_no_conflict
from ldp_experiment.graph_utils import EPS, add_edge_with_attrs, edge_by_id
from ldp_experiment.ldp_algorithm import run_ldp
from ldp_experiment.residual_ilp import solve_residual_circulation_ilp


def test_no_conflict_dp() -> None:
    cycles = [
        Cycle((1,), -8.0, 6, "neg_weight_pos_cost"),
        Cycle((2,), 3.0, -3, "nonneg_weight_neg_cost"),
        Cycle((3,), -2.0, 2, "neg_weight_pos_cost"),
        Cycle((4,), 1.0, -1, "nonneg_weight_neg_cost"),
        Cycle((5,), -1.0, 1, "neg_weight_pos_cost"),
    ]
    result = solve_by_knapsack_dp_no_conflict(cycles, B=3)
    selected = {c.edge_ids[0] for c in result.selected_cycles}
    assert result.improved
    assert result.total_cost <= 3
    assert {1, 2}.issubset(selected), "DP must keep a temporary over-budget item if later released"


def test_conflict_dp() -> None:
    cycles = [
        Cycle((1, 2), -5.0, 2, "neg_weight_pos_cost"),
        Cycle((2, 3), -6.0, 3, "neg_weight_pos_cost"),
        Cycle((4,), 2.0, -2, "nonneg_weight_neg_cost"),
        Cycle((5, 6), -4.0, 2, "neg_weight_pos_cost"),
        Cycle((6, 7), -3.0, 1, "neg_weight_pos_cost"),
    ]
    result = solve_by_conflict_dp(cycles, B=3)
    selected_sets = [set(c.edge_ids) for c in result.selected_cycles]
    for i in range(len(selected_sets)):
        for j in range(i + 1, len(selected_sets)):
            assert not (selected_sets[i] & selected_sets[j])
    assert result.objective <= -8.0 + EPS


def _small_residual() -> nx.MultiDiGraph:
    R = nx.MultiDiGraph()
    add_edge_with_attrs(R, "b", "a", weight=-5, cost=2, is_reverse=True, desc="anchor")
    add_edge_with_attrs(R, "a", "c", weight=1, cost=0, desc="path")
    add_edge_with_attrs(R, "c", "b", weight=1, cost=0, desc="path")
    add_edge_with_attrs(R, "d", "a", weight=-2, cost=-2, is_reverse=True, is_split_edge=True, desc="release anchor")
    add_edge_with_attrs(R, "a", "d", weight=3, cost=0, desc="return")
    add_edge_with_attrs(R, "a", "b", weight=10, cost=0, desc="discard positive")
    return R


def test_candidate_cycle_enum() -> None:
    R = _small_residual()
    cycles = enumerate_candidate_cycles(R)
    validate_candidate_cycles(R, cycles)
    kinds = {c.kind for c in cycles}
    assert "neg_weight_pos_cost" in kinds
    assert "nonneg_weight_neg_cost" in kinds
    assert len({frozenset(c.edge_ids) for c in cycles}) == len(cycles)


def test_ldp_residual_update() -> None:
    G = nx.MultiDiGraph()
    add_edge_with_attrs(G, "s", "a", weight=1, cost=0)
    add_edge_with_attrs(G, "a", "t", weight=1, cost=0)
    result = run_ldp(G, "s", "t", k=1, delta=1)
    assert result.feasible
    by_id = edge_by_id(result.residual)
    reverse_edges = [e for e in by_id.values() if e.is_reverse]
    assert len(reverse_edges) == 2
    assert all(e.weight == -1 for e in reverse_edges)
    assert any(e.is_split_edge and e.cost == 1 for e in by_id.values())


def test_ilp_if_available() -> None:
    R = _small_residual()
    cycles = enumerate_candidate_cycles(R)
    dp = solve_by_conflict_dp(cycles, B=2)
    ilp = solve_residual_circulation_ilp(R, B=2)
    if ilp.status != "NO_GUROBI":
        assert abs(dp.objective - ilp.objective) <= EPS


def run_all() -> None:
    test_no_conflict_dp()
    test_conflict_dp()
    test_candidate_cycle_enum()
    test_ldp_residual_update()
    test_ilp_if_available()
    print("all tests passed")


if __name__ == "__main__":
    run_all()
