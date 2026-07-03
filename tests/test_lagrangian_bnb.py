from __future__ import annotations

import random

import networkx as nx
import pytest

pytest.importorskip("gurobipy")

from ldp_experiment.graph_utils import EPS, add_edge_with_attrs
from ldp_experiment.lagrangian_bnb import solve_candidate_edge_lagrangian_bnb
from ldp_experiment.residual_ilp import solve_residual_circulation_ilp


def _empty_only_graph() -> nx.MultiDiGraph:
    R = nx.MultiDiGraph()
    add_edge_with_attrs(R, "a", "b", weight=2, cost=0)
    add_edge_with_attrs(R, "b", "a", weight=1, cost=0)
    return R


def _one_negative_feasible_cycle() -> nx.MultiDiGraph:
    R = nx.MultiDiGraph()
    add_edge_with_attrs(R, "a", "b", weight=-5, cost=2)
    add_edge_with_attrs(R, "b", "a", weight=1, cost=0)
    return R


def _over_budget_plus_release_graph() -> nx.MultiDiGraph:
    R = nx.MultiDiGraph()
    add_edge_with_attrs(R, "a", "b", weight=-10, cost=6)
    add_edge_with_attrs(R, "b", "a", weight=1, cost=0)
    add_edge_with_attrs(R, "c", "d", weight=1, cost=-4)
    add_edge_with_attrs(R, "d", "c", weight=0, cost=0)
    return R


def test_empty_solution_when_no_negative_feasible_circulation() -> None:
    result = solve_candidate_edge_lagrangian_bnb(_empty_only_graph(), B=3)
    assert result.status == "OPTIMAL"
    assert abs(result.objective) <= EPS
    assert not result.improved


def test_finds_one_budget_feasible_negative_cycle() -> None:
    result = solve_candidate_edge_lagrangian_bnb(_one_negative_feasible_cycle(), B=3)
    assert result.status == "OPTIMAL"
    assert result.objective < -EPS
    assert result.improved
    assert result.total_cost <= 3


def test_over_budget_negative_cycle_can_be_repaired_by_negative_cost_cycle() -> None:
    result = solve_candidate_edge_lagrangian_bnb(_over_budget_plus_release_graph(), B=3)
    assert result.status == "OPTIMAL"
    assert result.objective <= -9 + EPS
    assert result.total_cost <= 3
    assert result.improved


def test_matches_full_ilp_on_small_random_residual_graphs() -> None:
    rng = random.Random(7)
    for trial in range(5):
        R = nx.MultiDiGraph()
        R.add_nodes_from(range(4))
        for eid in range(7):
            u = rng.randrange(4)
            v = rng.randrange(4)
            while v == u:
                v = rng.randrange(4)
            add_edge_with_attrs(
                R,
                u,
                v,
                eid=eid,
                weight=rng.randint(-4, 5),
                cost=rng.randint(-2, 4),
            )
        full = solve_residual_circulation_ilp(R, B=3)
        bnb = solve_candidate_edge_lagrangian_bnb(
            R,
            B=3,
            candidate_eids=set(range(7)),
            max_lagrangian_iters=5,
        )
        assert full.status == "OPTIMAL"
        assert bnb.status == "OPTIMAL", f"trial={trial}, status={bnb.status}"
        assert abs(bnb.objective - full.objective) <= EPS


def test_exact_tail_with_tiny_candidate_set_still_matches_full_ilp() -> None:
    R = _one_negative_feasible_cycle()
    full = solve_residual_circulation_ilp(R, B=3)
    bnb = solve_candidate_edge_lagrangian_bnb(R, B=3, candidate_eids=set())
    assert full.status == "OPTIMAL"
    assert bnb.status == "OPTIMAL"
    assert bnb.num_exact_tail_calls == 1
    assert abs(bnb.objective - full.objective) <= EPS
